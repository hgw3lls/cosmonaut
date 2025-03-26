# 🧠 COSMONAUT  
**Infinite generative latent morphing engine**  
Smooth patch-based interpolation between evolving AI-generated visual prompts.

---

## ✨ Features

- 🔁 Infinite animation generation from evolving prompts
- 🎨 AI-assisted prompt chaining (OpenAI)
- 🧩 Patchwise latent morphing (grid-based)
- 🔀 Patch shuffling & motion trails
- 🌈 Morph strength gradients (radial, stripe, random)
- 🌀 Animated patch masks with blending (sin, perlin, radial, image)
- 🔧 CLI control over every parameter
- 💾 Save frames, videos, latent arrays, and prompt logs
- 🖼️ Real-time preview loop during generation (optional)

---

## 📦 Installation

```bash
git clone https://github.com/yourname/cosmonaut.git
cd cosmonaut
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Also create a `.env` file with:

```
REPLICATE_API_TOKEN=your_replicate_token
OPENAI_API_KEY=your_openai_key
```

---

## 🚀 Basic Usage

```bash
python cosmonaut_cli.py \
  --prompt_start "crystalline neural web" \
  --infinite_mode \
  --ai_chain_prompts \
  --output_dir ./outputs/neuralweb
```

---

## 🧠 Prompt Chaining (OpenAI)

```bash
--ai_chain_prompts
```

---

## 🧩 Patch Morphing

```bash
--patchwise_morph \
--patch_grid 6 \
--patch_ratio 0.8
```

- Gradient Morph: `--patch_strength_mode radial`
- Shuffle Patches: `--patch_shuffle_mode per_frame`
- Motion Trails: `--patch_motion drift`

---

## 🌀 Animated Patch Masks

```bash
--patch_mask_animated \
--patch_mask_type perlin \
--patch_mask_threshold 0.3
```

### Mask Blending:

```bash
--patch_mask_blend radial,perlin \
--patch_mask_blend_mode multiply
```

### Image-Based Mask:

```bash
--patch_mask_type image \
--patch_mask_image masks/my_mask.png
```

---

## 🎬 Interpolation Control

```bash
--transition_duration 10
```

or

```bash
--num_output_frames 480
```

---

## 🖼️ Output Options

```bash
--save_frames \
--save_montage \
--no_video \
--output_dir ./outputs/my_morph
```

---

## ✅ CLI Examples

### 1. Infinite Morph with AI Prompt + Patch Drift

```bash
python cosmonaut_cli.py \
  --prompt_start "liquid crystal membrane" \
  --infinite_mode \
  --ai_chain_prompts \
  --patchwise_morph \
  --patch_grid 6 \
  --patch_motion drift \
  --patch_strength_mode random \
  --patch_mask_animated \
  --patch_mask_type perlin \
  --patch_mask_threshold 0.25 \
  --transition_duration 10 \
  --save_frames \
  --output_dir ./outputs/perlin_crystal
```

### 2. Stripe Sweep with Radial Blend

```bash
python cosmonaut_cli.py \
  --prompt_start "neon circuitry fabric" \
  --infinite_mode \
  --ai_chain_prompts \
  --patchwise_morph \
  --patch_grid 8 \
  --patch_ratio 1.0 \
  --patch_mask_animated \
  --patch_mask_blend radial,stripe \
  --patch_mask_blend_mode max \
  --patch_mask_threshold 0.3 \
  --transition_duration 8 \
  --save_frames \
  --output_dir ./outputs/stripe_radial_blend
```

### 3. Static Mask + Stripe Gradient

```bash
python cosmonaut_cli.py \
  --prompt_start "folded iridescent tapestry" \
  --infinite_mode \
  --ai_chain_prompts \
  --patchwise_morph \
  --patch_grid 6 \
  --patch_strength_mode stripe \
  --patch_mask_animated \
  --patch_mask_type image \
  --patch_mask_image masks/ink_blot.png \
  --transition_duration 15 \
  --save_frames \
  --output_dir ./outputs/ink_morph
```

---

## 💬 Credits

Built with ❤️ by [YourName].  
Inspired by [tilemorph](https://github.com/andreasjansson/tile-morph), [tilemaker](https://github.com/replicate/tilemaker), and the latent creativity of neural networks.
