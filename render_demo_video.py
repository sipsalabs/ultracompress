"""Render the `uc demo` output as a 1080p MP4 terminal-style screen recording.

Approach: run `uc demo`, capture stdout in real time with timestamps, replay
into a pyte virtual terminal at video framerate, render each frame as a PNG
with monospace font on dark background, then stitch into MP4 with ffmpeg.

Output: ~60 sec, 1920×1080, H.264, ready to upload to YC's demo video field.
"""
from __future__ import annotations
import os, subprocess, time, threading, queue, shutil, tempfile, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ----- Config -----
TERM_COLS = 110
TERM_ROWS = 30
W, H = 1920, 1080
FPS = 30
FONT_SIZE = 20
PADDING = 40
CHAR_W = 0  # auto
LINE_H = 0  # auto
BG = (12, 14, 18)
FG = (220, 220, 220)
DIM = (130, 130, 130)
ACCENT = (102, 217, 239)  # cyan
GREEN = (102, 217, 102)
RED = (240, 100, 100)
YELLOW = (240, 220, 102)
WHITE = (245, 245, 245)
BLACK = (12, 14, 18)

# Pyte palette for ANSI 16-color
PALETTE = {
    'default': FG,
    'black': (40, 44, 52),
    'red': RED,
    'green': GREEN,
    'yellow': YELLOW,
    'blue': (100, 149, 237),
    'magenta': (199, 146, 234),
    'cyan': ACCENT,
    'white': WHITE,
    'brown': (180, 140, 80),
    'brightblack': (90, 96, 108),
    'brightred': (250, 130, 130),
    'brightgreen': (130, 240, 130),
    'brightyellow': (250, 235, 130),
    'brightblue': (130, 170, 250),
    'brightmagenta': (220, 170, 250),
    'brightcyan': (130, 230, 250),
    'brightwhite': (255, 255, 255),
}

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

# Try Cascadia Mono first, then Consolas as fallback
FONT_CANDIDATES = [
    "C:/Windows/Fonts/CascadiaMono.ttf",
    "C:/Windows/Fonts/CascadiaCode.ttf",
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
]
FONT_BOLD_CANDIDATES = [
    "C:/Windows/Fonts/CascadiaMonoBold.ttf",
    "C:/Windows/Fonts/CascadiaCodeBold.ttf",
    "C:/Windows/Fonts/consolab.ttf",
    "C:/Windows/Fonts/courbd.ttf",
]


def find_font(candidates):
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"None of these fonts found: {candidates}")


def capture_demo() -> list[tuple[float, bytes]]:
    """Run `uc demo` and capture stdout with timestamps. Returns list of (ts, chunk)."""
    print("Capturing uc demo output...")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TERM"] = "xterm-256color"
    env["COLUMNS"] = str(TERM_COLS)
    env["LINES"] = str(TERM_ROWS)
    env["FORCE_COLOR"] = "1"  # rich auto-detection
    env["COLORTERM"] = "truecolor"

    proc = subprocess.Popen(
        ["uc", "demo"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=0,
    )

    chunks = []
    t0 = time.time()
    while True:
        chunk = proc.stdout.read(64)
        if not chunk:
            break
        chunks.append((time.time() - t0, chunk))
    proc.wait()
    print(f"  captured {len(chunks)} chunks over {time.time() - t0:.1f}s")
    return chunks


def render_frames(chunks: list[tuple[float, bytes]], out_dir: Path, total_seconds: float | None = None):
    """Replay chunks into pyte and render frames."""
    import pyte

    screen = pyte.Screen(TERM_COLS, TERM_ROWS)
    stream = pyte.ByteStream(screen)

    if total_seconds is None:
        total_seconds = chunks[-1][0] + 1.5 if chunks else 5.0
    n_frames = int(total_seconds * FPS) + 1

    font_path = find_font(FONT_CANDIDATES)
    bold_path = find_font(FONT_BOLD_CANDIDATES)
    font = ImageFont.truetype(font_path, FONT_SIZE)
    bold = ImageFont.truetype(bold_path, FONT_SIZE)

    bbox = font.getbbox("M")
    char_w = bbox[2] - bbox[0]
    bbox = font.getbbox("Mg")
    line_h = int((bbox[3] - bbox[1]) * 1.4) + 2

    chunk_idx = 0
    print(f"Rendering {n_frames} frames at {FPS} fps...")

    for f in range(n_frames):
        t = f / FPS
        # Feed all chunks up to time t
        while chunk_idx < len(chunks) and chunks[chunk_idx][0] <= t:
            try:
                stream.feed(chunks[chunk_idx][1])
            except Exception:
                pass
            chunk_idx += 1

        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)

        # Draw header bar
        draw.rectangle([0, 0, W, 36], fill=(28, 32, 40))
        draw.ellipse([18, 12, 30, 24], fill=(255, 95, 86))
        draw.ellipse([38, 12, 50, 24], fill=(255, 189, 46))
        draw.ellipse([58, 12, 70, 24], fill=(39, 201, 63))
        draw.text((W // 2 - 80, 8), "ultracompress demo", fill=DIM, font=font)

        # Render each row
        cursor = screen.cursor
        for row_idx, line in enumerate(screen.display):
            x = PADDING
            y = PADDING + 36 + row_idx * line_h
            row_data = screen.buffer[row_idx]
            for col_idx in range(min(len(line), TERM_COLS)):
                ch = line[col_idx] if col_idx < len(line) else " "
                if ch == "\x00":
                    ch = " "
                ce = row_data.get(col_idx)
                fg_name = ce.fg if ce else "default"
                bgname = ce.bg if ce else "default"
                bold_flag = ce.bold if ce else False
                fg_color = PALETTE.get(fg_name, FG) if fg_name != "default" else FG
                f_use = bold if bold_flag else font
                if ch.strip():
                    draw.text((x, y), ch, fill=fg_color, font=f_use)
                x += char_w

        # Save frame
        img.save(out_dir / f"frame_{f:05d}.png", optimize=False, compress_level=1)
        if f % 60 == 0:
            print(f"  frame {f}/{n_frames} (t={t:.1f}s)")

    return n_frames, total_seconds


def main():
    out_dir = Path(tempfile.mkdtemp(prefix="uc_demo_render_"))
    try:
        chunks = capture_demo()
        if not chunks:
            print("ERROR: no output captured")
            sys.exit(1)
        n_frames, total = render_frames(chunks, out_dir)
        print(f"Rendered {n_frames} frames into {out_dir}")
        # Stitch with ffmpeg
        out_mp4 = Path(__file__).parent / "ultracompress_demo.mp4"
        cmd = [
            FFMPEG, "-y",
            "-framerate", str(FPS),
            "-i", str(out_dir / "frame_%05d.png"),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_mp4),
        ]
        print("Encoding MP4...")
        subprocess.run(cmd, check=True)
        size_mb = out_mp4.stat().st_size / 1024 / 1024
        print(f"Wrote {out_mp4} ({size_mb:.1f} MB, {total:.1f}s)")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
