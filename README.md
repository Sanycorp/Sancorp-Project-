# Sancorp — "NOT JUST AN MCP"

A **1920×1080, 60 fps, 16:9** motion-graphics title piece, generated entirely from
code. No After Effects, no stock footage, no external media — every pixel and
every sample is synthesised by Python.

It is a native-resolution reinterpretation of the reference reel in this repo
(`ai.akshu_1792149328308.mp4`, a 720×1280 / 30 fps phone recording of a laptop
screen): same subject — Claude driving After Effects over MCP — rebuilt as a
clean landscape master.

| | reference clip | this build |
|---|---|---|
| resolution | 720 × 1280 (9:16) | **1920 × 1080 (16:9)** |
| frame rate | 30 fps | **60 fps** |
| source | camera pointed at a screen | **rendered natively** |
| audio | recorded room audio | **procedural 120 BPM score, cut-synced** |
| duration | 91.5 s | 56 s |

Output: [`out/sancorp_mcp_1080p60.mp4`](out/sancorp_mcp_1080p60.mp4)

---

## The film

Seven scenes, locked to a 120 BPM grid (beat = 0.5 s, bar = 2 s) so every slam,
cut and flash lands on a transient in the score.

| # | time | scene | beat |
|---|------|-------|------|
| 1 | 0:00 – 0:08 | **Hook** | "THIS ISN'T AN MCP" slams in word-per-beat, then "IT'S A MOTION DESIGNER." |
| 2 | 0:08 – 0:18 | **The Bridge** | Claude → MCP → After Effects node diagram with travelling packets and a live tool-call log |
| 3 | 0:18 – 0:26 | **The Prompt** | terminal types a prompt, the build streams back in 3.1 s |
| 4 | 0:26 – 0:34 | **Kinetic Type** | knockout-disc typography, then "60 FPS / ZERO RENDER QUEUE" |
| 5 | 0:34 – 0:42 | **The Punchline** | a render bar that crawls, then collapses a 3-hour ETA to zero |
| 6 | 0:42 – 0:48 | **Output** | six live mini-compositions popping on 16ths, converging to centre |
| 7 | 0:48 – 0:56 | **Outro** | SANCORP wordmark build + spec plate |

---

## How it is made

```
src/
  engine.py    canvas, easing, text/shape cache, blending, post chain
  widgets.py   cards, chips, HUD, window chrome, progress bars, vector glyphs
  scenes.py    the seven scenes + the timeline
  audio.py     the synthesiser and the arrangement
  render.py    frame assembly, post, parallel encode
  fonts/       Anton · Archivo Black · Bebas Neue · Inter · JetBrains Mono
```

### Rendering

A frame is a `float32` numpy buffer `(1080, 1920, 3)` in 0..1. Vector art is
drawn with Pillow into RGBA layers and composited with normal / additive
blending, then pushed through a fixed post chain:

```
scene draw → slice glitch → zoom + shake → chromatic aberration
           → bloom → vignette + scanlines → HUD → flash → grain → filmic tonemap
```

Three optimisations keep it to ~0.3 s/frame on two cores:

* **Bounding-box compositing** — `over()` / `add()` crop each layer to its
  non-empty alpha box before touching numpy, which is ~9× faster than blending
  full-frame buffers for typical sparse layers.
* **Strided bloom downsample** — a 2-tap `[::4]` decimation instead of a box
  mean (~35× faster) ahead of two Gaussian passes at quarter resolution.
* **Seamless grid fields** — the dotted background is a pre-multiplied float
  tile of `(H+step, W+step)`; drifting it is a free strided *view* rather than a
  roll or a re-draw.

Encoding is split into eight segments rendered in parallel, each piped as raw
RGB24 straight into its own `libx264 -preset slow -crf 16` process, then
stream-copied together — so the pixels are only ever compressed once.

### Score

`audio.py` is a small additive synthesiser (no scipy). Kick, clap, hats,
bass, plucks and pads are built from summed harmonics with per-partial decay;
noise elements are shaped in the frequency domain with `rfft`. The arrangement
follows the cut list — half-time under the quiet scenes, risers into every
cut, a 3.4 s impact on the render pay-off — and the music bus is side-chain
ducked against the kick grid.

---

## Rebuilding

Requires Python 3.11+, `numpy`, `pillow`, and `imageio-ffmpeg` (which ships a
static ffmpeg binary).

```bash
pip install numpy pillow imageio-ffmpeg

cd src
python3 audio.py  ../out/score.wav                       # ~4 s
python3 render.py --video ../out/sancorp_mcp_1080p60.mp4 \
                  --audio ../out/score.wav --crf 16 --workers 2
```

Useful while iterating:

```bash
python3 render.py --probe 2.8 16.2 28.6   # single PNG stills at those seconds
python3 render.py --sheet                 # 5×5 contact sheet of the whole film
```

Everything is deterministic — grain and glitch are seeded per frame index, so
re-rendering any frame reproduces it exactly.
