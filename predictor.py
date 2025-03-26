from generation.local import generate_from_local, denoise
from generation.replicate import generate_from_replicate
from generation.interpolate import interpolate_latents
from prompts.chainer import chain_prompt_ai
# from ui.preview import start_realtime_preview_loop
from video import save_video, save_frames_to_png
from config import RESOLUTION_MAP
from utils import prompt_to_seed, decode_latent

import os
from typing import Iterator
from PIL import Image

class Predictor:
	def __init__(self, replicate_model="tstramer/material-diffusion:a42692c54c0f407f803a0a8a9066160976baedb77c91171a01730f9b0d7beeff"):
		self.tileable = False
		self.replicate_model = replicate_model
		self.pipe = None

	def setup(self):
		from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
		import torch

		if self.tileable:
			from utils import enable_tileable_conv
			enable_tileable_conv()

		print("🛰️ Loading local diffusion model...")
		self.pipe = DiffusionPipeline.from_pretrained(
			"runwayml/stable-diffusion-v1-5",
			torch_dtype=torch.float32,
			cache_dir="diffusers-cache",
		)
		self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
		self.pipe.to("cpu")

	def predict(self, **kwargs):
		if kwargs.get("transition_duration") and not kwargs.get("num_interpolation_steps"):
			# fps = kwargs.get("preview_fps") or kwargs.get("frames_per_second") or 20
			kwargs["num_interpolation_steps"] = int(fps * kwargs["transition_duration"])
			print(f"🌀 Using transition_duration={kwargs['transition_duration']}s → "
				  f"{kwargs['num_interpolation_steps']} interpolation steps at {fps} FPS")

		return self.predict_infinite(**kwargs) if kwargs.get("infinite_mode") else self.predict_finite(**kwargs)

	def predict_finite(self, **kwargs) -> Iterator[str]:
		prompts = kwargs.get("prompts", [])
		resolution = kwargs.get("resolution", "512x512")
		if resolution not in RESOLUTION_MAP:
			raise ValueError(f"Unsupported resolution '{resolution}'")
		width, height = RESOLUTION_MAP[resolution]

		seeds = [prompt_to_seed(p) for p in prompts]
		latents = []
		for i, prompt in enumerate(prompts):
			latent = generate_from_replicate(self.pipe, prompt, width, height, kwargs["num_inference_steps"], self.replicate_model) \
				if kwargs.get("use_replicate") else \
				generate_from_local(self.pipe, prompt, seeds[i], width, height, kwargs["num_inference_steps"], kwargs["noise_scale"], kwargs.get("quiet"))
			latents.append(latent)

		frames = interpolate_latents(latents, kwargs.get("num_interpolation_steps", 10), kwargs.get("looping_mode", False),
									 False, kwargs.get("tile_rows", 1), kwargs.get("tile_cols", 1), kwargs.get("quiet", False))

		images = [decode_latent(self.pipe, latent) for latent in frames]
		save_frames_to_png(images, out_dir=okwargs["output_dir"])
		video_path = save_video(images, width, height, kwargs["frames_per_second"], os.path.join(kwargs["output_dir"], "output"))
		yield video_path

	def predict_infinite(self, **kwargs) -> Iterator:
		prompt = kwargs.get("prompt_start", "A beautiful abstract pattern")
		output_dir = kwargs.get("output_dir", "output")
		os.makedirs(output_dir, exist_ok=True)
		# preview_dir = os.path.join(output_dir, "frames")
		# os.makedirs(preview_dir, exist_ok=True)
		prompt_log_path = os.path.join(output_dir, "prompt_log.txt")
		with open(prompt_log_path, "w") as f:
			f.write("")

		# if kwargs.get("show_preview"):
			# start_realtime_preview_loop(preview_dir, delay=int(1000 / kwargs.get("preview_fps", 20)))

		prev_prompt = prompt
		seed = prompt_to_seed(prompt)
		prev_latent = generate_from_replicate(self.pipe, prev_prompt, 512, 512, kwargs["num_inference_steps"], self.replicate_model) \
			if kwargs.get("use_replicate") else \
			generate_from_local(self.pipe, prev_prompt, seed, 512, 512, kwargs["num_inference_steps"], kwargs["noise_scale"], False)

		frame_counter = 0

		while True:
			new_prompt = chain_prompt_ai(prev_prompt, frame_counter) if kwargs.get("ai_chain_prompts") else prev_prompt + f" {frame_counter}"
			with open(prompt_log_path, "a") as f:
				f.write(f"[{frame_counter:06d}] {new_prompt}\n")

			seed = prompt_to_seed(new_prompt)
			new_latent = generate_from_replicate(self.pipe, new_prompt, 512, 512, kwargs["num_inference_steps"], self.replicate_model) \
				if kwargs.get("use_replicate") else \
				generate_from_local(self.pipe, new_prompt, seed, 512, 512, kwargs["num_inference_steps"], kwargs["noise_scale"], False)

			steps = kwargs.get("num_interpolation_steps", 10)
			frames = interpolate_latents([prev_latent, new_latent], steps, False, False, 1, 1, False)

			for frame in frames:
				image = decode_latent(self.pipe, frame)
				frame_path = os.path.join(output_dir, f"frame_{frame_counter:06d}.png")
				Image.fromarray(image).save(frame_path)
				frame_counter += 1
				yield image

			prev_latent = new_latent
			prev_prompt = new_prompt
