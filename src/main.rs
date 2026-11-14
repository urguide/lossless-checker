// Fake-lossless detector — analyze one file or a whole library to flag audio that is not the
// genuine lossless it claims to be.
//
// Three signals, all derived from the decoded PCM (the file's container/metadata is never
// trusted):
//   1. High-frequency cutoff cliff — lossy transcodes (MP3/AAC) low-pass at a fixed Hz, leaving
//      almost no energy above it. Genuine lossless extends naturally toward ~20kHz.
//   2. Sample-rate authenticity — a hi-res (>48k) file whose real content walls at the CD
//      Nyquist (~22kHz) is CD/lossy material upsampled into a hi-res container.
//   3. Spectral holes — AAC/Vorbis-style notches (report-only; too false-positive-prone to judge on).
//
// This is a heuristic (classical/vocal/old recordings naturally have little HF and may false-
// positive); its value is batch-flagging the highly suspicious. It is strictly read-only.
//
// Two subcommands:
//   `check`     — the PCM fake-lossless detector above (FLAC/ALAC/WAV/…). DSD is not handled here.
//   `check-dsd` — native DSD authenticity check: parses .dsf/.dff itself, measures the noise-shaping
//                 spectrum of the raw 1-bit stream (no ffmpeg). See `mod dsd`.

mod decode;
mod dsd;
mod i18n;
mod report;
mod safety;
mod spectrum;
mod verdict;

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::sync::atomic::{AtomicUsize, Ordering};

use clap::{Parser, Subcommand, ValueEnum};
use rayon::prelude::*;

use decode::decode_audio;
use dsd::judge::DsdThresholds;
use dsd::{run_check_dsd, run_export_spectrum, ChannelSel, DsdCheckArgs, ExportArgs};
use i18n::Lang;
use report::{build_json, build_text_report, Outcome};
use spectrum::SpectrumOpts;
use verdict::{classify, Analysis};

#[derive(Parser)]
#[command(name = "lossless-checker")]
#[command(about = "Detect fake-lossless audio (PCM transcodes/upsampled hi-res) and fake DSD", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Detect fake-lossless PCM (lossy transcodes / upsampled hi-res). Scans a file or folder.
    Check(CheckArgs),
    /// Detect fake DSD (PCM/lossy "washed" into DSD) via native noise-shaping analysis (no ffmpeg).
    CheckDsd(CheckDsdArgs),
    /// Export one DSD file's power spectrum to CSV (frequency_hz,power_db) for plotting.
    ExportSpectrum(ExportSpectrumArgs),
}

#[derive(Parser)]
struct CheckArgs {
    /// Path to a single audio file, or a directory to scan recursively
    path: PathBuf,

    /// Peak-relative cutoff threshold, in dB below the spectral peak. This is the default detection
    /// method: the cutoff is the highest frequency whose energy stays within this many dB of the
    /// track's own peak. Calibrated to 65.0; override only for debugging.
    #[arg(long, default_value_t = 65.0)]
    peak_db: f64,

    /// Use the legacy noise-floor detection method (with --threshold) instead of peak-relative.
    #[arg(long)]
    noise_floor: bool,

    /// Noise-floor multiplier for the legacy method (only used with --noise-floor).
    #[arg(long, default_value_t = 10.0)]
    threshold: f64,

    /// Write the text report to this file instead of stdout (directory scan only)
    #[arg(long)]
    report: Option<PathBuf>,

    /// Also write a JSON report to this file (directory scan only)
    #[arg(long)]
    json: Option<PathBuf>,

    /// Comma-separated extensions to scan (lossless/PCM containers; mp3 etc. are pointless here, and
    /// DSD goes through the `check-dsd` subcommand)
    #[arg(long, default_value = "flac,wav,m4a,aif,aiff,caf,alac")]
    ext: String,

    /// Number of parallel worker threads, 1..=32 (default: CPU cores capped at 4)
    #[arg(long)]
    jobs: Option<usize>,

    /// Output language for logs and reports: zh (中文, default) or en. JSON is unaffected.
    #[arg(long, value_enum, default_value = "zh")]
    lang: Lang,
}

#[derive(Parser)]
struct CheckDsdArgs {
    /// One or more .dsf/.dff files or directories (directories are scanned recursively)
    paths: Vec<PathBuf>,

    /// FFT size for the Welch power spectrum (DSD64 @65536 ≈ 43 Hz/bin)
    #[arg(long, default_value_t = 65536)]
    fft_size: usize,

    /// Noise-shaping slope fit: lower bound in Hz at DSD64 (scaled up with the DSD rate)
    #[arg(long, default_value_t = 30_000.0)]
    slope_lo: f64,

    /// Noise-shaping slope fit: upper bound in Hz at DSD64 (scaled up with the DSD rate)
    #[arg(long, default_value_t = 100_000.0)]
    slope_hi: f64,

    /// Minimum noise-shaping slope (dB/oct) below which a file is flagged
    #[arg(long, default_value_t = 6.0)]
    min_slope: f64,

    /// Ultrasonic energy is summed above this frequency (Hz) at DSD64 (scaled up with the DSD rate)
    #[arg(long, default_value_t = 50_000.0)]
    hf_threshold: f64,

    /// Minimum ultrasonic energy ratio below which a file is flagged
    #[arg(long, default_value_t = 0.05)]
    min_hf_ratio: f64,

    /// Number of parallel worker threads, 1..=32 (default: CPU cores capped at 4)
    #[arg(long, short = 'j')]
    jobs: Option<usize>,

    /// Output format: text (default) or json
    #[arg(long, value_enum, default_value = "text")]
    format: OutputFormat,

    /// Also print a per-album aggregation
    #[arg(long)]
    album_summary: bool,

    /// Print metrics for every file, not just the suspicious ones
    #[arg(long, short = 'v')]
    verbose: bool,

    /// Output language for logs and the text report: zh (中文, default) or en. JSON is unaffected.
    #[arg(long, value_enum, default_value = "zh")]
    lang: Lang,
}

#[derive(Parser)]
struct ExportSpectrumArgs {
    /// The .dsf/.dff file to analyze
    file: PathBuf,

    /// FFT size for the Welch power spectrum
    #[arg(long, default_value_t = 65536)]
    fft_size: usize,

    /// CSV output path [default: <file>.spectrum.csv]
    #[arg(long, short = 'o')]
    output: Option<PathBuf>,

    /// Channel to export: `mix` (all channels averaged, default) or a 0-based index
    #[arg(long, default_value = "mix")]
    channel: String,

    /// Output language for log messages: zh (中文, default) or en
    #[arg(long, value_enum, default_value = "zh")]
    lang: Lang,
}

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

fn main() {
    match Cli::parse().command {
        Command::Check(args) => run_check(args),
        Command::CheckDsd(args) => run_dsd(args),
        Command::ExportSpectrum(args) => run_export(args),
    }
}

/// `check` subcommand: the PCM fake-lossless detector.
fn run_check(cli: CheckArgs) {
    if !cli.peak_db.is_finite()
        || !(1.0..=120.0).contains(&cli.peak_db)
        || !cli.threshold.is_finite()
        || cli.threshold <= 0.0
    {
        eprintln!("invalid detection threshold");
        exit(2);
    }
    configure_jobs(cli.jobs);

    // None = legacy noise-floor method (with --threshold); Some(db) = peak-relative (default).
    let peak_db = if cli.noise_floor {
        None
    } else {
        Some(cli.peak_db)
    };

    if cli.path.is_file() {
        // --report / --json only apply to directory scans; warn rather than silently drop them.
        if cli.report.is_some() || cli.json.is_some() {
            eprintln!(
                "{}",
                cli.lang.pick(
                    "提示：--report / --json 仅用于目录扫描，单文件模式下已忽略。",
                    "Note: --report / --json apply to directory scans only; ignored in single-file mode."
                )
            );
        }
        run_single(&cli.path, cli.threshold, peak_db, cli.lang);
    } else if cli.path.is_dir() {
        run_batch(&cli);
    } else {
        eprintln!(
            "{}: {}",
            cli.lang.pick(
                "路径不存在或无法访问",
                "path does not exist or is not accessible"
            ),
            safety::display(&cli.path)
        );
        exit(1);
    }
}

/// `check-dsd` subcommand: native DSD authenticity check.
fn run_dsd(cli: CheckDsdArgs) {
    configure_jobs(cli.jobs);

    let thresholds = DsdThresholds {
        min_noise_shaping_slope: cli.min_slope,
        min_hf_ratio: cli.min_hf_ratio,
        slope_fit_lo_hz: cli.slope_lo,
        slope_fit_hi_hz: cli.slope_hi,
        hf_threshold_hz: cli.hf_threshold,
        ..Default::default()
    };

    let code = run_check_dsd(DsdCheckArgs {
        paths: cli.paths,
        fft_size: cli.fft_size,
        thresholds,
        as_json: matches!(cli.format, OutputFormat::Json),
        album_summary: cli.album_summary,
        verbose: cli.verbose,
        lang: cli.lang,
    });
    exit(code);
}

/// `export-spectrum` subcommand: dump a single file's power spectrum to CSV.
fn run_export(cli: ExportSpectrumArgs) {
    configure_jobs(None);
    let channel = match cli.channel.trim().to_ascii_lowercase().as_str() {
        "mix" => ChannelSel::Mix,
        other => match other.parse::<usize>() {
            Ok(i) => ChannelSel::Index(i),
            Err(_) => {
                eprintln!(
                    "{}",
                    cli.lang.pick(
                        "--channel 只接受 `mix` 或一个 0 起始的声道序号",
                        "--channel accepts only `mix` or a 0-based channel index"
                    )
                );
                exit(2);
            }
        },
    };

    let code = run_export_spectrum(ExportArgs {
        file: cli.file,
        fft_size: cli.fft_size,
        output: cli.output,
        channel,
        lang: cli.lang,
    });
    exit(code);
}

fn configure_jobs(jobs: Option<usize>) {
    let jobs = jobs.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(4)
    });
    if !(1..=32).contains(&jobs) {
        eprintln!("--jobs must be in 1..=32");
        exit(2);
    }
    if let Err(e) = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build_global()
    {
        eprintln!("failed to initialize workers: {e}");
        exit(1);
    }
}

/// Build the detection options from CLI knobs.
fn opts(threshold: f64, peak_db: Option<f64>) -> SpectrumOpts {
    SpectrumOpts {
        peak_db,
        threshold_mult: threshold,
    }
}

/// Single-file mode: print the detailed per-file verdict.
fn run_single(path: &Path, threshold: f64, peak_db: Option<f64>, lang: Lang) {
    let decoded = match decode_audio(path, lang) {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "{}: {}",
                lang.pick("解码失败", "decode failed"),
                safety::escape(&e)
            );
            exit(1);
        }
    };

    if decoded.sample_count == 0 {
        eprintln!(
            "{}",
            lang.pick("没有解码出任何采样", "no samples were decoded")
        );
        exit(1);
    }

    let features = spectrum::analyze_energy(
        &decoded.energy,
        decoded.sample_rate,
        opts(threshold, peak_db),
    );
    let nyquist = decoded.sample_rate as f32 / 2.0;
    let ratio = if nyquist > 0.0 {
        features.cutoff_hz / nyquist
    } else {
        0.0
    };
    let verdict = classify(&features, decoded.sample_rate);

    println!("{}: {}", lang.pick("文件", "File"), safety::display(path));
    println!("{}: {}", lang.pick("格式", "Format"), decoded.format_label);
    println!(
        "{}: {} Hz",
        lang.pick("采样率", "Sample rate"),
        decoded.sample_rate
    );
    println!(
        "{}: {}",
        lang.pick("采样总数", "Total samples"),
        decoded.sample_count
    );
    println!(
        "{}: {:.0} Hz",
        lang.pick("奈奎斯特频率", "Nyquist frequency"),
        nyquist
    );
    println!(
        "{}: {:.0} Hz ({:.1}% of Nyquist)",
        lang.pick("估计高频截止", "Estimated HF cutoff"),
        features.cutoff_hz,
        ratio * 100.0
    );
    if let Some(db) = features.hires_ext_db {
        println!(
            "{}: {db:.1} dB ({})",
            lang.pick("高频延伸(>26kHz)", "HF extension (>26kHz)"),
            lang.pick(
                "相对频谱峰值；越低代表高频越空",
                "relative to spectral peak; lower = emptier highs"
            )
        );
    }
    if features.holes.is_empty() {
        println!(
            "{}",
            lang.pick("频谱空洞: 无明显空洞", "Spectral holes: none significant")
        );
    } else {
        println!(
            "{}: {} {}",
            lang.pick("频谱空洞", "Spectral holes"),
            features.holes.len(),
            lang.pick(
                "处（仅供参考，不影响判定）",
                "(informational only, does not affect the verdict)"
            )
        );
        for h in &features.holes {
            match lang {
                Lang::Zh => {
                    println!(
                        "  - {:.0}-{:.0} Hz（深约 {:.0} dB）",
                        h.low_hz, h.high_hz, h.depth_db
                    )
                }
                Lang::En => {
                    println!(
                        "  - {:.0}-{:.0} Hz (~{:.0} dB deep)",
                        h.low_hz, h.high_hz, h.depth_db
                    )
                }
            }
        }
    }
    println!();
    println!(
        "{}: {}",
        lang.pick("判断", "Verdict"),
        verdict.sentence(lang)
    );
    println!();
    println!(
        "{}",
        lang.pick(
            "（提示：古典乐、人声、老录音本身高频能量就少，可能误报；",
            "(Note: classical, vocal and old recordings naturally have little HF and may false-positive;"
        )
    );
    println!(
        "{}",
        lang.pick(
            "  建议对可疑文件用 Spek 等工具看一眼频谱图再下结论。）",
            "  for suspect files, eyeball the spectrum in a tool like Spek before concluding.)"
        )
    );
}

/// Directory mode: scan in parallel, then write the text report (and optional JSON).
fn run_batch(cli: &CheckArgs) {
    let exts: Vec<String> = cli
        .ext
        .split(',')
        .map(|e| e.trim().trim_start_matches('.').to_ascii_lowercase())
        .filter(|e| !e.is_empty())
        .collect();

    let lang = cli.lang;
    eprintln!(
        "{} ({}: {})",
        lang.pick("收集音频文件中…", "Collecting audio files…"),
        lang.pick("扩展名", "extensions"),
        exts.join(", ")
    );
    let mut files = match collect_audio_files(&cli.path, &exts) {
        Ok(files) => files,
        Err(e) => {
            eprintln!("cannot completely enumerate input directory: {e}");
            exit(1);
        }
    };
    files.sort();

    if files.is_empty() {
        eprintln!(
            "{} {}",
            lang.pick(
                "在以下目录未找到匹配的音频文件:",
                "no matching audio files found under"
            ),
            safety::display(&cli.path)
        );
        exit(1);
    }

    let total = files.len();
    eprintln!(
        "{}",
        lang.pick(
            &format!("找到 {total} 个文件，开始扫描…"),
            &format!("Found {total} files, scanning…")
        )
    );

    let done = AtomicUsize::new(0);
    let threshold = cli.threshold;
    let peak_db = if cli.noise_floor {
        None
    } else {
        Some(cli.peak_db)
    };
    let mut outcomes: Vec<Outcome> = files
        .par_iter()
        .map(|p| {
            let o = analyze_file(p, threshold, peak_db, lang);
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_multiple_of(25) || n == total {
                eprint!("\r{} {n}/{total}", lang.pick("扫描中…", "Scanning…"));
                let _ = io::stderr().flush();
            }
            o
        })
        .collect();
    eprintln!();
    // Deterministic ordering for the report regardless of thread completion order.
    outcomes.sort_by(|a, b| a.path.cmp(&b.path));

    let text = build_text_report(&cli.path, &outcomes, lang);
    match &cli.report {
        Some(p) => match safety::create_output(p).and_then(|mut f| f.write_all(text.as_bytes())) {
            Ok(()) => eprintln!(
                "{}: {}",
                lang.pick("文本报告已写入", "Text report written to"),
                safety::display(p)
            ),
            Err(e) => {
                eprintln!(
                    "{} ({}): {e}",
                    lang.pick("写文本报告失败", "failed to write text report"),
                    safety::display(p)
                );
                exit(1);
            }
        },
        None => print!("{text}"),
    }

    if let Some(p) = &cli.json {
        let json = build_json(&cli.path, &outcomes);
        let serialized = match serde_json::to_string_pretty(&json) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "{}: {e}",
                    lang.pick("序列化 JSON 失败", "failed to serialize JSON")
                );
                exit(1);
            }
        };
        match safety::create_output(p).and_then(|mut f| f.write_all(serialized.as_bytes())) {
            Ok(()) => eprintln!(
                "{}: {}",
                lang.pick("JSON 报告已写入", "JSON report written to"),
                safety::display(p)
            ),
            Err(e) => {
                eprintln!(
                    "{} ({}): {e}",
                    lang.pick("写 JSON 报告失败", "failed to write JSON report"),
                    safety::display(p)
                );
                exit(1);
            }
        }
    }
    if outcomes.iter().any(|o| o.result.is_err()) {
        exit(1);
    }
}

/// Iterative traversal, without following links or silently omitting unreadable directories.
fn collect_audio_files(root: &Path, exts: &[String]) -> io::Result<Vec<PathBuf>> {
    if !std::fs::symlink_metadata(root)?.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "scan root must be a non-symlink directory",
        ));
    }
    let mut out = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    let mut visited = 0usize;
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            visited += 1;
            if visited > 1_000_000 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "scan exceeds one million entries; split the library",
                ));
            }
            let path = entry.path();
            let kind = entry.file_type()?;
            if kind.is_dir() {
                pending.push(path);
            } else if kind.is_file() {
                let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
                    continue;
                };
                if exts.iter().any(|want| want == &ext.to_ascii_lowercase()) {
                    out.push(path);
                }
            }
        }
    }
    Ok(out)
}

/// Decode + analyze one file. Decoder errors are surfaced in the report.
fn analyze_file(path: &Path, threshold: f64, peak_db: Option<f64>, lang: Lang) -> Outcome {
    let result = (|| {
        let decoded = decode_audio(path, lang)?;
        if decoded.sample_count == 0 {
            return Err(lang
                .pick("没有解码出任何采样", "no samples were decoded")
                .to_string());
        }
        let features = spectrum::analyze_energy(
            &decoded.energy,
            decoded.sample_rate,
            opts(threshold, peak_db),
        );
        let nyquist = decoded.sample_rate as f32 / 2.0;
        Ok(Analysis {
            sample_rate: decoded.sample_rate,
            format_label: decoded.format_label,
            cutoff_hz: features.cutoff_hz,
            ratio: if nyquist > 0.0 {
                features.cutoff_hz / nyquist
            } else {
                0.0
            },
            hole_count: features.holes.len(),
            hires_ext_db: features.hires_ext_db,
            verdict: classify(&features, decoded.sample_rate),
        })
    })();
    Outcome {
        path: path.to_path_buf(),
        result,
    }
}
