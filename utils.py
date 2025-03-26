import hashlib
import torch
import cv2
import numpy as np

def patch_conv(**patch):
	"""
	Patch the initialization of torch.nn.Conv2d to use custom keyword arguments.
	
	This is useful, for example, to force the convolutional layers to use circular padding,
	which can help create seamlessly tileable images.
	
	Usage:
		patch_conv(padding_mode="circular")
	"""
	cls = torch.nn.Conv2d
	init = cls.__init__
	def __init__(self, *args, **kwargs):
		for k, v in patch.items():
			kwargs[k] = v
		return init(self, *args, **kwargs)
	cls.__init__ = __init__

def enable_tileable_conv():
	"""
	Enable tileable convolution by patching torch.nn.Conv2d to use circular padding.
	"""
	patch_conv(padding_mode="circular")

def prompt_to_seed(prompt: str) -> int:
	"""
	Convert a text prompt to a deterministic integer seed.
	
	Uses SHA-256 hashing and takes the first few characters to produce an integer.
	
	Args:
		prompt: A string prompt.
	
	Returns:
		An integer seed.
	"""
	return int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:4], 16)

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
	"""
	Perform spherical linear interpolation (slerp) between two latent vectors.
	
	This function computes the interpolation along the shortest path on a hypersphere,
	which can produce smooth transitions when the latent vectors are normalized.
	
	Args:
		t: Interpolation parameter (float between 0 and 1).
		v0: Starting latent vector (torch.Tensor).
		v1: Ending latent vector (torch.Tensor).
		DOT_THRESHOLD: If the cosine of the angle between vectors is greater than this,
					   the function falls back to linear interpolation.
	
	Returns:
		Interpolated latent vector (torch.Tensor).
	"""
	# Normalize the vectors
	v0_norm = v0 / v0.norm(dim=-1, keepdim=True)
	v1_norm = v1 / v1.norm(dim=-1, keepdim=True)
	dot = (v0_norm * v1_norm).sum(-1)
	
	# If the vectors are nearly parallel, use linear interpolation to avoid numerical issues.
	if dot.abs().mean() > DOT_THRESHOLD:
		return (1 - t) * v0 + t * v1

	theta_0 = torch.acos(dot)
	sin_theta_0 = torch.sin(theta_0)
	theta_t = theta_0 * t
	s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
	s1 = torch.sin(theta_t) / sin_theta_0
	return s0.unsqueeze(-1) * v0 + s1.unsqueeze(-1) * v1

def decode_latent(pipe, latent):
	with torch.no_grad():
		if isinstance(latent, np.ndarray):
			latent = torch.from_numpy(latent).to(torch.float32)
		elif isinstance(latent, torch.Tensor):
			latent = latent.to(torch.float32)
		decoded = pipe.decode_latents(latent)
		if isinstance(decoded, torch.Tensor):
			decoded = decoded.detach().cpu().permute(0, 2, 3, 1).numpy()
		image = (decoded[0] * 255).round().astype(np.uint8)
		return image
