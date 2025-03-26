import numpy as np
from tqdm import tqdm
from utils import slerp
from perlin_noise import PerlinNoise
from PIL import Image

def create_patch_shuffle_map(patch_grid):
	total = patch_grid * patch_grid
	flat_indices = list(range(total))
	np.random.shuffle(flat_indices)
	return np.array(flat_indices).reshape((patch_grid, patch_grid))

def generate_patch_mask(mask_type, patch_grid, j, steps, speed, threshold, mask_image_path=None):
	y, x = np.mgrid[0:patch_grid, 0:patch_grid]
	if mask_type == "sin":
		return np.sin(speed * (j / steps) + 0.3 * x + 0.5 * y)
	elif mask_type == "stripe":
		return np.sin(speed * (j / steps) + x)
	elif mask_type == "radial":
		cx, cy = patch_grid / 2, patch_grid / 2
		r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
		return np.sin(speed * (j / steps) + r * 0.5)
	elif mask_type == "image" and mask_image_path:
		img = Image.open(mask_image_path).convert("L").resize((patch_grid, patch_grid))
		return np.array(img) / 255.0
	elif mask_type == "perlin":
		noise_gen = PerlinNoise(octaves=4, seed=42)
		return np.array([
			[noise_gen([(iy + speed * (j / steps)) / patch_grid,
						(ix + speed * (j / steps)) / patch_grid])
			 for ix in range(patch_grid)]
			for iy in range(patch_grid)
		])
	else:
		return np.random.rand(patch_grid, patch_grid)

def interpolate_latents(
	latents, steps, looping, circular, tile_rows, tile_cols, quiet,
	patchwise=False, patch_grid=4, patch_ratio=0.3, patch_shuffle_mode="none",
	patch_motion="none", patch_strength_mode="uniform",
	patch_mask_animated=False, patch_mask_speed=0.5, patch_mask_threshold=0.4,
	patch_mask_type="sin", patch_mask_blend="", patch_mask_blend_mode="multiply",
	patch_mask_image=""
):
	frames = []

	if circular and len(latents) >= 2:
		A, B = latents[0], latents[1]
		for i in tqdm(range(steps * len(latents)), desc="Circular Interp", disable=quiet):
			theta = 2 * np.pi * (i / (steps * len(latents)))
			L = A * np.cos(theta) + B * np.sin(theta)
			frames.append(L)
	else:
		for i in range(len(latents) if looping else len(latents) - 1):
			a, b = latents[i], latents[(i + 1) % len(latents)]

			shuffle_map = None
			if patchwise and patch_shuffle_mode == "per_transition":
				shuffle_map = create_patch_shuffle_map(patch_grid)

			for j in range(steps):
				t = j / steps
				if patchwise and patch_shuffle_mode == "per_frame":
					shuffle_map = create_patch_shuffle_map(patch_grid)

				mask = None
				if patchwise and patch_mask_animated:
					types = (patch_mask_blend.split(",") if patch_mask_blend else [patch_mask_type])
					masks = [generate_patch_mask(t.strip(), patch_grid, j, steps, patch_mask_speed, patch_mask_threshold, patch_mask_image) for t in types]

					if len(masks) == 1:
						combined = masks[0]
					else:
						if patch_mask_blend_mode == "multiply":
							combined = masks[0] * masks[1]
						elif patch_mask_blend_mode == "add":
							combined = np.clip(masks[0] + masks[1], 0, 1)
						elif patch_mask_blend_mode == "min":
							combined = np.minimum(masks[0], masks[1])
						elif patch_mask_blend_mode == "max":
							combined = np.maximum(masks[0], masks[1])
						else:
							combined = masks[0]
					mask = (combined > patch_mask_threshold).astype(np.float32)

				if patchwise:
					frame = patchwise_slerp(
						t, a, b,
						num_patches=patch_grid,
						interpolate_ratio=patch_ratio,
						shuffle_map=shuffle_map,
						motion=patch_motion,
						motion_frame=j,
						motion_steps=steps,
						strength_mode=patch_strength_mode,
						active_mask=mask
					)
				else:
					frame = slerp(t, a, b)
				frames.append(frame)

	return frames

def patchwise_slerp(
	t, a, b,
	num_patches=4,
	interpolate_ratio=0.25,
	shuffle_map=None,
	motion="none",
	motion_frame=0,
	motion_steps=10,
	strength_mode="uniform",
	active_mask=None
):
	_, C, H, W = a.shape
	patch_H, patch_W = H // num_patches, W // num_patches
	out = a.clone()

	strengths = np.ones((num_patches, num_patches), dtype=np.float32)

	if strength_mode == "random":
		strengths = np.random.rand(num_patches, num_patches)
	elif strength_mode == "radial":
		for i in range(num_patches):
			for j in range(num_patches):
				dx = i - num_patches // 2
				dy = j - num_patches // 2
				dist = np.sqrt(dx ** 2 + dy ** 2)
				strengths[i, j] = 1.0 - (dist / (np.sqrt(2) * num_patches / 2))
	elif strength_mode == "stripe":
		for i in range(num_patches):
			strengths[i, :] = i / num_patches

	for i in range(num_patches):
		for j in range(num_patches):
			if active_mask is not None:
				if active_mask[i, j] < 0.5:
					continue
			else:
				if np.random.rand() > interpolate_ratio:
					continue

			y0, y1 = i * patch_H, (i + 1) * patch_H
			x0, x1 = j * patch_W, (j + 1) * patch_W

			si, sj = i, j
			if shuffle_map is not None:
				flat = shuffle_map[i, j]
				si, sj = divmod(flat, num_patches)

			dx, dy = 0, 0
			if motion == "drift":
				drift_strength = 0.5
				angle = 2 * np.pi * (motion_frame / motion_steps)
				dx = int(np.round(np.sin(angle) * drift_strength * patch_W))
				dy = int(np.round(np.cos(angle) * drift_strength * patch_H))

			sy0, sy1 = si * patch_H + dy, (si + 1) * patch_H + dy
			sx0, sx1 = sj * patch_W + dx, (sj + 1) * patch_W + dx

			sy0, sy1 = np.clip([sy0, sy1], 0, H)
			sx0, sx1 = np.clip([sx0, sx1], 0, W)

			patch_a = a[:, :, y0:y1, x0:x1]
			patch_b = b[:, :, sy0:sy1, sx0:sx1]

			min_h = min(patch_a.shape[2], patch_b.shape[2])
			min_w = min(patch_a.shape[3], patch_b.shape[3])
			patch_a = patch_a[:, :, :min_h, :min_w]
			patch_b = patch_b[:, :, :min_h, :min_w]

			strength = strengths[i, j]
			scaled_t = t * strength
			morphed = slerp(scaled_t, patch_a, patch_b)
			out[:, :, y0:y0 + min_h, x0:x0 + min_w] = morphed

	return out
