# `uc demo`

Play a scripted demo session for screen recording or local exploration. No Hugging Face Hub access required.

## Synopsis

```
uc demo [--speed FLOAT] [--no-pause]
```

## Options

| Option | Default | Description |
|---|---|---|
| `--speed FLOAT` | `1.0` | Playback speed multiplier (e.g., `0.5` for half speed, `2.0` for double speed) |
| `--no-pause` | off | Skip inter-step pauses (useful for non-interactive recording) |

## What it does

Runs a 60-90 second scripted simulation of typical UltraCompress CLI usage:

- Banner display
- A simulated `uc list` showing pre-compressed models
- A simulated `uc pull` with download progress
- A simulated `uc info` showing the manifest
- A simulated `uc bench` showing benchmark output

The demo does **not** call out to the Hugging Face Hub or any external service. All output is simulated locally with realistic timing. Useful for:

- Demos at conferences without reliable Wi-Fi
- Screen recordings for marketing materials
- Quick exploration of what the CLI looks like before installing
- CI smoke tests

## Examples

```bash
# Standard playback at real-time speed
uc demo

# Fast playback for non-interactive recording
uc demo --speed 2.0 --no-pause

# Slow playback for live walk-through
uc demo --speed 0.5
```

## Recording the demo

For a clean recording for marketing / launch:

```bash
# Linux / macOS — using script(1)
script -t 2>demo.timing demo.txt
uc demo --no-pause
exit
# Then: ttyrec / asciinema for the recording

# Or just use OBS Studio / QuickTime to record the terminal window
```

For a 60-second YouTube cut:

1. Open terminal at 1080p, font ~16pt, dark background
2. Record full window with OBS at 30fps
3. `uc demo --no-pause`
4. Trim to ~60 seconds in your editor
5. Add subtitles + brand outro

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Demo played to completion |
| 0 | Terminal interruption (Ctrl-C) — exits cleanly |
| 2 | Invalid arguments (Click default) |

## What this command is NOT

- **Not a real benchmark.** `uc demo`'s output is fixed scripted content; numbers are illustrative.
- **Not a fake CLI.** The command is honest about what it is — a *scripted demo*, not real usage. We won't ship `--demo` flags that pretend to be real.

For a real demo, run actual commands in sequence:

```bash
uc list
uc pull sipsalabs/<model-id>
uc info ./models/sipsalabs_<model-id>
uc bench ./models/sipsalabs_<model-id> --tasks hellaswag --limit 100
```

## See also

- [Quickstart](../getting-started/quickstart.md) — what to actually run
- [`uc list`](list.md), [`uc pull`](pull.md), [`uc info`](info.md), [`uc bench`](bench.md) — the real commands
