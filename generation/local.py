# generation/local.py

import torch
from tqdm import tqdm

def generate_from_local(pipe, prompt, seed, width, height, steps, noise_scale, quiet):
	generator = torch.Generator("cpu").manual_seed(seed)
	noise = torch.randn((1, pipe.unet.in_channels, height // 8, width // 8), generator=generator) * noise_scale
	text_emb = pipe._encode_prompt(prompt, "cpu", 1, True, "")
	pipe.scheduler.set_timesteps(steps)
	return denoise(pipe, noise, text_emb, steps, 7.5, generator, quiet=quiet)

def denoise(pipe, latents, text_embeddings, steps, guidance_scale, generator, quiet=False):
	scheduler = pipe.scheduler
	extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0)
	for t in tqdm(scheduler.timesteps, desc="Denoising", disable=quiet):
		latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
		latent_input = scheduler.scale_model_input(latent_input, t)
		noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
		if guidance_scale > 1.0:
			uncond, cond = noise_pred.chunk(2)
			noise_pred = uncond + guidance_scale * (cond - uncond)
		latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
	return latents
