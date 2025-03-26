import numpy as np
from tqdm import tqdm
from utils import slerp

def interpolate_latents(latents, steps, looping, circular, tile_rows, tile_cols, quiet, patchwise=False):
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
			for j in range(steps):
				t = j / steps
				if patchwise:
					frame = patchwise_slerp(t, a, b, num_patches=patch_grid, interpolate_ratio=patch_ratio)
				else:
					frame = slerp(t, a, b)
				frames.append(frame)
	return frames

def patchwise_slerp(t, a, b, num_patches=4, interpolate_ratio=0.25):
	_, C, H, W = a.shape
	patch_H, patch_W = H // num_patches, W // num_patches
	out = a.clone()

	for i in range(num_patches):
		for j in range(num_patches):
			if np.random.rand() < interpolate_ratio:
				y0, y1 = i * patch_H, (i + 1) * patch_H
				x0, x1 = j * patch_W, (j + 1) * patch_W
				patch_a = a[:, :, y0:y1, x0:x1]
				patch_b = b[:, :, y0:y1, x0:x1]
				out[:, :, y0:y1, x0:x1] = slerp(t, patch_a, patch_b)
	return out
