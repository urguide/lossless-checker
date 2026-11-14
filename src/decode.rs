//! Bounded streaming decode and per-channel power-spectrum accumulation.
//!
//! symphonia handles the lossless/PCM containers (FLAC, ALAC, WAV, AIFF, CAF, …) with no system
//! dependencies. DSD (.dsf/.dff) has no symphonia codec and is handled by the separate `check-dsd`
//! subcommand (native 1-bit analysis, see `crate::dsd`), not here.

use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::i18n::Lang;

/// Decoded result: mean channel power spectrum + sample rate +
/// a human-readable source-format label for the reports.
pub struct DecodedAudio {
    pub energy: Vec<f64>,
    pub sample_count: u64,
    pub sample_rate: u32,
    pub format_label: String,
}

/// Stream an audio file into a power spectrum. DSD has a separate subcommand.
pub fn decode_audio(path: &Path, lang: Lang) -> Result<DecodedAudio, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    if matches!(ext.as_deref(), Some("dsf") | Some("dff")) {
        return Err(lang
            .pick(
                "DSD 文件请用 check-dsd 子命令分析（check 仅处理 PCM/无损容器）",
                "DSD files: use the `check-dsd` subcommand (`check` handles PCM/lossless only)",
            )
            .to_string());
    }
    decode_symphonia(path, lang)
}

/// Decode with symphonia, keeping channel powers separate until averaging.
fn decode_symphonia(path: &Path, lang: Lang) -> Result<DecodedAudio, String> {
    let mut file = crate::safety::open_input(path)
        .map_err(|e| format!("{}: {e}", lang.pick("打不开文件", "cannot open file")))?;
    let aiff_frames = aiff_frame_count(&mut file).map_err(|e| format!("invalid AIFF: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Use the file extension as a hint to help the probe identify the format.
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| {
            format!(
                "{}: {e}",
                lang.pick("无法识别音频格式", "unrecognized audio format")
            )
        })?;

    let mut format = probed.format;

    let track = format.default_track().ok_or_else(|| {
        lang.pick("找不到音频轨", "no audio track found")
            .to_string()
    })?;
    let track_id = track.id;
    let expected_frames = aiff_frames.or(track.codec_params.n_frames);

    let format_label = symphonia::default::get_codecs()
        .get_codec(track.codec_params.codec)
        .map(|d| d.short_name.to_uppercase())
        .unwrap_or_else(|| "PCM".to_string());

    use symphonia::core::codecs::*;
    if matches!(
        track.codec_params.codec,
        CODEC_TYPE_AAC
            | CODEC_TYPE_MP1
            | CODEC_TYPE_MP2
            | CODEC_TYPE_MP3
            | CODEC_TYPE_VORBIS
            | CODEC_TYPE_ADPCM_IMA_WAV
            | CODEC_TYPE_ADPCM_MS
    ) {
        return Err(format!(
            "lossy codec detected: {format_label}; not a lossless container"
        ));
    }

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions { verify: true })
        .map_err(|e| {
            format!(
                "{}: {e}",
                lang.pick("无法创建解码器", "cannot create decoder")
            )
        })?;

    // Decode the whole track (no early stop — see spectrum::avg_power_spectrum for why).
    let mut spectrum = None;
    let mut sample_count = 0u64;
    let mut sample_rate: u32 = track.codec_params.sample_rate.unwrap_or(0);
    let mut channels: usize = 0;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(e) => return Err(format!("packet read failed: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                let spec = *audio_buf.spec();
                if sample_rate == 0 {
                    sample_rate = spec.rate;
                }
                let new_channels = spec.channels.count();
                if !(1..=8).contains(&new_channels) || !(8_000..=768_000).contains(&spec.rate) {
                    return Err("unsupported channel count or sample rate".into());
                }
                if spec.rate != sample_rate || (channels != 0 && channels != new_channels) {
                    return Err("audio format changes mid-stream".into());
                }
                channels = new_channels;
                if audio_buf.capacity() > 1_048_576 {
                    return Err("decoded packet exceeds sample limit".into());
                }

                let mut sample_buf = SampleBuffer::<f32>::new(audio_buf.capacity() as u64, spec);
                sample_buf.copy_interleaved_ref(audio_buf);
                sample_count += (sample_buf.samples().len() / channels) as u64;
                spectrum
                    .get_or_insert_with(|| crate::spectrum::StreamingSpectrum::new(channels))
                    .push(sample_buf.samples())?;
            }
            Err(e) => return Err(format!("{}: {e}", lang.pick("解码出错", "decode error"))),
        }
    }

    if sample_rate == 0 {
        return Err(lang
            .pick("无法确定采样率", "could not determine sample rate")
            .to_string());
    }
    if sample_rate < 38_000 {
        return Err("inconclusive: sample rate is too low for the 19 kHz cutoff heuristic".into());
    }

    if expected_frames.is_some_and(|n| sample_count < n) {
        return Err(format!(
            "truncated audio: decoded {sample_count} frames, declared {}",
            expected_frames.unwrap()
        ));
    }
    if decoder.finalize().verify_ok == Some(false) {
        return Err("audio integrity verification failed".into());
    }
    Ok(DecodedAudio {
        energy: spectrum.ok_or("no samples decoded")?.finish()?,
        sample_count,
        sample_rate,
        format_label,
    })
}

/// Symphonia 0.5's AIFF demuxer includes the SSND offset/block header in n_frames.
/// Read the authoritative COMM frame count, validating physical chunk bounds first.
fn aiff_frame_count(file: &mut std::fs::File) -> std::io::Result<Option<u64>> {
    use std::io::{Error, ErrorKind, Read, Seek, SeekFrom};
    let len = file.metadata()?.len();
    if len < 12 {
        return Ok(None);
    }
    let mut header = [0u8; 12];
    file.read_exact(&mut header)?;
    file.rewind()?;
    if &header[..4] != b"FORM" || !matches!(&header[8..], b"AIFF" | b"AIFC") {
        return Ok(None);
    }
    let end = u32::from_be_bytes(header[4..8].try_into().unwrap()) as u64 + 8;
    if end > len || end < 12 {
        return Err(Error::new(ErrorKind::UnexpectedEof, "invalid FORM length"));
    }
    let mut pos = 12u64;
    let mut count = None;
    let mut chunks = 0;
    while pos < end {
        chunks += 1;
        if chunks > 100_000 || end - pos < 8 {
            return Err(Error::new(ErrorKind::InvalidData, "invalid AIFF chunks"));
        }
        file.seek(SeekFrom::Start(pos))?;
        let mut chunk = [0u8; 8];
        file.read_exact(&mut chunk)?;
        let size = u32::from_be_bytes(chunk[4..].try_into().unwrap()) as u64;
        let next = pos + 8 + size + size % 2;
        if next > end {
            return Err(Error::new(ErrorKind::UnexpectedEof, "truncated AIFF chunk"));
        }
        if &chunk[..4] == b"COMM" {
            if size < 18 || count.is_some() {
                return Err(Error::new(ErrorKind::InvalidData, "invalid COMM"));
            }
            let mut fields = [0u8; 6];
            file.read_exact(&mut fields)?;
            count = Some(u32::from_be_bytes(fields[2..].try_into().unwrap()) as u64);
        }
        pos = next;
    }
    file.rewind()?;
    count
        .map(Some)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "missing COMM"))
}
