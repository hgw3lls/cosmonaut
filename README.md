# 🛰️ Cosmonaut - Infinite Latent Morph Generator

A modular pipeline for generating abstract, tileable, and infinitely evolving animations using AI prompt chaining, latent interpolation, and real-time preview.

Built with `diffusers`, OpenAI, and Replicate. Inspired by [tile-morph](https://github.com/andreasjansson/tile-morph) and [tilemaker](https://github.com/replicate/tilemaker).

---

## 📦 Features

- 🔁 Infinite animation mode
- 🎨 AI-driven prompt chaining via OpenAI
- 🧠 Local and Replicate-based inference
- 🧱 Latent interpolation (linear, spherical, bilinear)
- 🖼️ Real-time GUI preview (with Tkinter + OpenCV)
- 🧩 Grid/tileable texture output
- 💾 Frame and video export

---

## 🛠️ Installation

```bash
git clone https://github.com/hgw3lls/cosmonaut
cd cosmonaut
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You’ll also need:

- A `.env` file with:
  ```
  OPENAI_API_KEY=your-key-here
  REPLICATE_API_TOKEN=your-token-here
  ```

---

## 🚀 Usage

### 🌀 Infinite AI prompt chaining mode (tileable textures):

```bash
python cosmist_cli.py \
  --prompt_start "abstract tessellated geometry" \
  --infinite_mode \
  --ai_chain_prompts \
  --use_replicate \
  --tileable \
  --transition_duration 10 \
  --save_frames \
  --output_dir ./outputs/infinite
```

### 🎞️ Finite morph animation between list of prompts:

```bash
python cosmist_cli.py \
  --prompt_list_path prompts.txt \
  --num_interpolation_steps 30 \
  --resolution 512x512 \
  --save_frames \
  --output_dir ./outputs/finite
```

---

## 🔧 CLI Options (highlights)

| Option                  | Description                                 |
|------------------------|---------------------------------------------|
| `--infinite_mode`       | Run endlessly using prompt chaining         |
| `--ai_chain_prompts`    | Use OpenAI to refine and evolve prompts     |
| `--prompt_list_path`    | Provide a .txt file with static prompts     |
| `--use_replicate`       | Use Replicate model instead of local        |
| `--tileable`            | Enables seamless circular padding           |
| `--save_frames`         | Saves individual images as PNGs             |
| `--transition_duration` | Seconds per morph (e.g. `--transition_duration 12`) |

---

## 📁 Output Structure

```bash
outputs/
└── infinite/
    ├── frames/
    │   ├── frame_000000.png
    │   ├── ...
    ├── prompt_log.txt
    └── output.mp4  # if in finite mode
```

---

## 🧩 Modular Breakdown

```bash
cosmonaut/
├── cosmist_cli.py              # CLI runner
├── predictor.py                # Core dispatcher
├── generation/
│   ├── local.py                # Local latent generation
│   ├── replicate.py            # Replicate API backend
│   └── interpolate.py          # Latent interpolation
├── prompts/
│   └── chainer.py              # AI prompt evolution
├── ui/
│   └── preview.py              # Live GUI animation
├── utils/
│   ├── image.py                # Decode/display helpers
│   └── tiling.py               # Circular conv patching
```

---

## 🧠 Ideas & Next Features

- 🧬 Latent loop optimizations
- 🔂 Hotkey to pause/resume infinite generation
- 🌐 Web dashboard interface
- 📦 Plugin-style model swapping

---

## 🪐 Credit

Originally based on [@andreasjansson](https://github.com/andreasjansson)'s tile-morph concept and Replicate's tilemaker idea — now modular, infinite, and AI-chained ✨
