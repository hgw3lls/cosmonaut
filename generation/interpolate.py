import numpy as np
from tqdm import tqdm
from utils import slerp

def create_patch_shuffle_map(patch_grid):
	total = patch_grid * patch_grid
	flat_indices = list(range(total))
	np.random.shuffle(flat_indices)
	return np.array(flat_indices).reshape((patch_grid, patch_grid))	


def interpolate_latents(
		latents, steps, looping, circular, tile_rows, tile_cols, quiet,
		patchwise=False, patch_grid=4, patch_ratio=0.3, patch_shuffle_mode="none",
		patch_motion="none", patch_strength_mode="uniform",
		patch_mask_animated=False, patch_mask_speed=0.5, patch_mask_threshold=0.4
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
	
				# Precompute shuffle map if per_transition
				shuffle_map = None
				if patchwise and patch_shuffle_mode == "per_transition":
					shuffle_map = create_patch_shuffle_map(patch_grid)
	
				for j in range(steps):
					t = j / steps
					
					if patchwise and patch_shuffle_mode == "per_frame":
						shuffle_map = create_patch_shuffle_map(patch_grid)
					
					if patchwise:
						mask = None
						if patchwise and patch_mask_animated:
							# Animated sinusoidal mask
							y, x = np.mgrid[0:patch_grid, 0:patch_grid]
							noise = np.sin(
								patch_mask_speed * (j / steps) + 0.3 * x + 0.5 * y
							)
							mask = (noise > patch_mask_threshold).astype(np.float32)
					
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
		
			for i in range(num_patches):
				for j in range(num_patches):
					if active_mask is not None:
						if active_mask[i, j] < 0.5:
							continue
					else:
						if np.random.rand() > interpolate_ratio:
							continue
		
					# Base patch indices
					y0, y1 = i * patch_H, (i + 1) * patch_H
					x0, x1 = j * patch_W, (j + 1) * patch_W
		
					# Shuffled patch destination
					si, sj = i, j
					if shuffle_map is not None:
						flat = shuffle_map[i, j]
						si, sj = divmod(flat, num_patches)
		
					# Motion trail offsets
					dx, dy = 0, 0
					if motion == "drift":
						drift_strength = 0.5  # Fraction of patch size
						angle = 2 * np.pi * (motion_frame / motion_steps)
						dx = int(np.round(np.sin(angle) * drift_strength * patch_W))
						dy = int(np.round(np.cos(angle) * drift_strength * patch_H))
		
					# Calculate offset target patch in B
					sy0, sy1 = si * patch_H + dy, (si + 1) * patch_H + dy
					sx0, sx1 = sj * patch_W + dx, (sj + 1) * patch_W + dx
		
					# Clamp to image bounds
					sy0, sy1 = np.clip([sy0, sy1], 0, H)
					sx0, sx1 = np.clip([sx0, sx1], 0, W)
		
					# Extract patches
					patch_a = a[:, :, y0:y1, x0:x1]
					patch_b = b[:, :, sy0:sy1, sx0:sx1]
		
					# Handle size mismatch from clipping
					min_h = min(patch_a.shape[2], patch_b.shape[2])
					min_w = min(patch_a.shape[3], patch_b.shape[3])
					patch_a = patch_a[:, :, :min_h, :min_w]
					patch_b = patch_b[:, :, :min_h, :min_w]
		
					out[:, :, y0:y0 + min_h, x0:x0 + min_w] = slerp(t, patch_a, patch_b)
		
			return out