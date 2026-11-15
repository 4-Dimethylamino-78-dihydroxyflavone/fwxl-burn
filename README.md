# fwxl-burn

A terminal UI that transcribes audio/video and burns the resulting subtitles
into the video. It orchestrates [Faster-Whisper-XXL] and ffmpeg, and runs two
speech-recognition engines of its own in-process on the GPU (NVIDIA Parakeet
and Meta Omnilingual).

One self-contained binary, no installer, no required config file.

[Faster-Whisper-XXL]: https://github.com/Purfview/whisper-standalone-win

## What it does

- **Transcribe** a file or a whole queue to SRT, with several Whisper presets
  (VAD, diarization and vocal-extraction variants), Parakeet TDT, or the
  1600-language Omnilingual CTC model.
- **Hardburn** an SRT into the video with a hardware encoder and a choice of
  subtitle styles.
- **Reuse** embedded or existing subtitles instead of transcribing.
- **Never lose a finished transcription**: an unusable run is rebuilt from
  sibling formats, retried on reduced settings after an out-of-memory failure,
  or preserved in a `.fwxl-recovery` folder.
- **Resume** a queue: files already transcribed with the same mode and model
  are skipped.

## Platform support

| | Windows x64 | Linux x64 | macOS |
|---|---|---|---|
| TUI, queueing, subtitle tools | yes | yes | yes |
| Whisper modes (W E Q I U Y) | yes | yes¹ | no² |
| Parakeet / Omnilingual (P O) | yes, NVIDIA | yes, NVIDIA | no³ |
| Hardburn | NVENC | NVENC | VideoToolbox |

¹ Needs the Linux build of Faster-Whisper-XXL.
² Faster-Whisper-XXL has no macOS build.
³ These engines are CUDA-only. They fail with a clear error rather than
falling back to the CPU.

Day-to-day development and testing happens on Windows with an NVIDIA GPU.
Linux and macOS are built and unit-tested in CI; the macOS VideoToolbox
hardburn has not been exercised on real hardware yet.

**This is a GPU tool by design.** There is no CPU fallback anywhere: a run that
cannot reach the GPU stops and says why, instead of quietly taking ten times
longer.

## Requirements

- **ffmpeg and ffprobe** (required). Any recent build with the `subtitles`
  filter and your platform's hardware encoder.
- **Faster-Whisper-XXL** (optional). Needed only for the Whisper modes.
- **CUDA 13 + cuDNN 9 runtime** (optional). Needed only for Parakeet and
  Omnilingual. ONNX Runtime documents CUDA 13.2 / cuDNN 9.23 as the minimum;
  CUDA 13.0 with cuDNN 9.19 has been verified to work. CUDA 12 is no longer
  supported. The libraries only need to be on the loader path (`PATH` on
  Windows, `LD_LIBRARY_PATH` on Linux); a full toolkit install is not
  required.

## Install

Build from source with [rustup](https://rustup.rs):

```bash
cargo build --release
```

Any current stable Rust works; the minimum supported version is 1.92.
`rust-toolchain.toml` selects `stable`, so rustup picks it up automatically.

The binary is `target/release/fwxl-burn` (`fwxl-burn.exe` on Windows). The
ONNX Runtime provider libraries that the build places next to it
(`onnxruntime_providers_*`) must stay beside the binary.

On Linux the build additionally needs `pkg-config` and `libssl-dev` (used only
while downloading ONNX Runtime at build time; the finished binary does not
link OpenSSL).

## Run

```bash
fwxl-burn                      # open the TUI
fwxl-burn video.mkv other.mp4  # open the TUI with files queued
fwxl-burn --headless --input video.mkv --mode U
fwxl-burn --init               # write an editable fwxl-burn.toml
```

Files can also be pasted or dropped into the start screen, one or many at a
time.

## Finding the tools

Nothing needs configuring if the tools are discoverable. For each tool left
unset in the config, fwxl-burn searches, in order:

1. beside the fwxl-burn executable, and a `Faster-Whisper-XXL` folder there
2. beside the config file, and a `Faster-Whisper-XXL` folder there
3. Subtitle Edit's `Purfview-Whisper-Faster` folder (faster-whisper-xxl only)
4. `PATH`

So the two zero-config layouts are "ffmpeg is on `PATH`" and "portable
folder":

```
fwxl-burn/
├── fwxl-burn(.exe)
└── Faster-Whisper-XXL/
    ├── faster-whisper-xxl(.exe)
    ├── ffmpeg(.exe)
    └── ffprobe(.exe)
```

A path you set explicitly under `[tools]` must exist. It is never silently
replaced by another copy found elsewhere.

## Configuration

Configuration is optional. Without a config file the defaults compiled into
the binary are used. `fwxl-burn --init` writes those defaults to
`fwxl-burn.toml` for editing; every setting is documented inline there (see
[config.default.toml](config.default.toml)).

Config files are looked up in this order:

1. `--config <path>`
2. `fwxl-burn.toml` beside the executable
3. the per-user config directory:
   `%APPDATA%\fwxl-burn\config.toml` (Windows),
   `~/.config/fwxl-burn/config.toml` (Linux),
   `~/Library/Application Support/fwxl-burn/config.toml` (macOS)
4. `fwxl-burn.toml` in the working directory

## Development

```bash
cargo test
```

The test suite is hermetic. Two tests that need a GPU and local media are
`#[ignore]`d; see the comments on them for how to run them.

## License

MIT. See [LICENSE](LICENSE).
