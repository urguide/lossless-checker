"""Black-box regression tests using tiny generated fixtures, never real audio libraries."""
import array
import json
import math
import os
import pathlib
import random
import struct
import subprocess
import sys
import tempfile
import wave


BIN = pathlib.Path(sys.argv[1]).resolve()


def run(*args, ok=True):
    p = subprocess.run([str(BIN), *map(str, args)], capture_output=True, timeout=15)
    assert (p.returncode == 0) == ok, (args, p.returncode, p.stdout, p.stderr)
    assert p.returncode >= 0 and b"panicked" not in p.stderr, (args, p.stderr)
    return p


def wav(path, samples, channels=1, rate=44100):
    values = array.array("h", samples)
    if sys.byteorder != "little":
        values.byteswap()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(values.tobytes())


def dsf(payload, channels=1, block=4096, rate=2822400, samples=None):
    if samples is None:
        samples = len(payload) * 8 // channels
    return (b"DSD " + struct.pack("<QQQ", 28, 92 + len(payload), 0)
            + b"fmt " + struct.pack("<QIIIIIIQII", 52, 1, 0, 1, channels, rate, 1, samples, block, 0)
            + b"data" + struct.pack("<Q", 12 + len(payload)) + payload)


def chunk(tag, body):
    return tag + struct.pack(">Q", len(body)) + body + b"\0" * (len(body) % 2)


def dff(prop, payload=b""):
    body = b"DSD " + chunk(b"PROP", prop) + chunk(b"DSD ", payload)
    return b"FRM8" + struct.pack(">Q", len(body)) + body


def main():
    # Build fixtures only in the repository's ignored target directory.
    target = pathlib.Path("target")
    target.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="security-", dir=target) as tmp:
        root = pathlib.Path(tmp)
        library = root / "library"
        library.mkdir()
        tone = [round(24000 * math.sin(2 * math.pi * 1000 * i / 44100)) for i in range(16384)]
        source = library / "tone.wav"
        wav(source, tone)
        original = source.read_bytes()
        run("check", source)
        # AIFF COMM is authoritative: the decoder's SSND-derived frame count includes 8 header bytes.
        pcm_be = b"".join(struct.pack(">h", x) for x in tone)
        comm = struct.pack(">HIH", 1, len(tone), 16) + bytes.fromhex("400eac44000000000000")
        aiff_body = b"AIFF" + b"COMM" + struct.pack(">I", len(comm)) + comm
        sound = b"\0" * 8 + pcm_be
        aiff_body += b"SSND" + struct.pack(">I", len(sound)) + sound
        aiff = root / "valid.aiff"
        aiff.write_bytes(b"FORM" + struct.pack(">I", len(aiff_body)) + aiff_body)
        run("check", aiff)
        (root / "truncated.aiff").write_bytes(aiff.read_bytes()[:-10])
        run("check", root / "truncated.aiff", ok=False)
        wav(root / "short.wav", tone[:100])
        wav(root / "silent.wav", [0] * 16384)
        wav(root / "low-rate.wav", tone, rate=16000)
        for name in ["short.wav", "silent.wav", "low-rate.wav"]:
            run("check", root / name, ok=False)

        # Equal and opposite stereo channels must retain their spectral evidence.
        stereo = library / "antiphase.wav"
        wav(stereo, [v for x in tone for v in (x, -x)], channels=2)
        report = root / "report.json"
        run("check", library, "--json", report)
        results = json.loads(report.read_text())["results"]
        assert len(results) == 2
        assert results[0]["verdict"] == results[1]["verdict"] == "suspect"
        assert abs(results[0]["cutoff_hz"] - results[1]["cutoff_hz"]) <= 1

        # All output paths reject existing files, including the input itself.
        run("check", library, "--report", source, ok=False)
        run("check", library, "--json", source, ok=False)
        assert source.read_bytes() == original
        report_bytes = report.read_bytes()
        run("check", library, "--json", report, ok=False)
        assert report.read_bytes() == report_bytes

        truncated = root / "truncated.wav"
        truncated.write_bytes(original[:-4096])
        run("check", truncated, ok=False)
        bad_library = root / "broken"
        bad_library.mkdir()
        (bad_library / "bad.wav").write_bytes(b"not audio")
        run("check", bad_library, ok=False)
        floats = struct.pack("<f", float("nan")) * 8192
        body = b"WAVE" + b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, 44100, 176400, 4, 32)
        body += b"data" + struct.pack("<I", len(floats)) + floats
        (root / "nan.wav").write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
        run("check", root / "nan.wav", ok=False)

        if os.name == "posix":
            os.mkfifo(library / "pipe.flac")
            (library / "linked.wav").symlink_to(source.resolve())
            run("check", library)
            run("check", library / "linked.wav", ok=False)
            link = root / "output-link"
            link.symlink_to(source.resolve())
            run("check", library, "--report", link, ok=False)
            hardlink = root / "output-hardlink"
            os.link(source, hardlink)
            run("check", library, "--json", hardlink, ok=False)
            assert source.read_bytes() == original
            hostile = library / "bad\x1b[2J\nFAKE.wav"
            hostile.write_bytes(original)
            output = run("check", library).stdout
            assert b"\x1b" not in output and b"\\nFAKE" in output

        payload = random.Random(7).randbytes(65536)
        valid = root / "valid.dsf"
        valid.write_bytes(dsf(payload))
        run("check-dsd", valid)
        csv = root / "spectrum.csv"
        run("export-spectrum", valid, "-o", csv)
        assert csv.read_text().startswith("frequency_hz,power_db")
        original_dsf = valid.read_bytes()
        run("export-spectrum", valid, "-o", valid, ok=False)
        assert valid.read_bytes() == original_dsf
        for fft in ["0", "1", "9999", "4294967295"]:
            run("check-dsd", valid, "--fft-size", fft, ok=False)
            run("export-spectrum", valid, "--fft-size", fft, ok=False)
        for args in [("check", source, "--peak-db", "NaN"),
                     ("check", source, "--jobs", "0"),
                     ("check-dsd", valid, "--min-hf-ratio", "NaN")]:
            run(*args, ok=False)

        attacks = {
            "huge-block.dsf": dsf(b"", block=0xffffffff),
            "huge-channels.dsf": dsf(b"", channels=0xffffffff),
            "zero-rate.dsf": dsf(payload, rate=0),
            "truncated.dsf": dsf(payload)[:-8],
            "count-mismatch.dsf": dsf(payload, samples=1),
            "overflow.dff": dff(b"SND " + b"FS  " + struct.pack(">Q", 0xffffffffffffffff)),
            "huge-prop.dff": b"FRM8" + struct.pack(">Q", 16) + b"DSD PROP" + struct.pack(">Q", 0xffffffffffffffff),
            "truncated.dff": dff(b"SND " + chunk(b"FS  ", struct.pack(">I", 2822400))
                                  + chunk(b"CHNL", struct.pack(">H", 1)), payload)[:-8],
        }
        for name, content in attacks.items():
            path = root / name
            path.write_bytes(content)
            run("check-dsd", path, ok=False)
        print("Security regressions passed: malformed DSD, output preservation, special files, PCM integrity, stereo, CLI limits")


if __name__ == "__main__":
    main()
