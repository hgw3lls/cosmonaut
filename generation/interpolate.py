# generation/interpolate.py

from tqdm import tqdm
import numpy as np
from utils import slerp

def interpolate_latents(latents, steps, looping, circular, tile_rows, tile_cols, quiet):
	frames = []
	if circular and len(latents) >= 2:
		A, B = latents[0], latents[1]
		for i in tqdm(range(steps * len(latents)), desc="Circular Interp", disable=quiet):
			theta = 2 * np.pi * (i / (steps * len(latents)))
			frames.append(A * np.cos(theta) + B * np.sin(theta))
	else:
		for i in range(len(latents) if looping else len(latents) - 1):
			a, b = latents[i], latents[(i + 1) % len(latents)]
			for j in range(steps):
				t = j / steps
				frame = slerp(t, a, b)
				frames.append(frame)
	return frames
