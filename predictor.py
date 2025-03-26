from generation.local import generate_from_local, denoise
from generation.replicate import generate_from_replicate
from generation.interpolate import interpolate_latents
from prompts.chainer import chain_prompt_ai
# from ui.preview import start_realtime_preview_loop
from video import save_video, save_frames_to_png
from config import RESOLUTION_MAP
from utils import prompt_to_seed, decode_latent
import numpy as np
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
		from generation.local import generate_from_local
		from generation.replicate import generate_from_replicate
		from generation.interpolate import interpolate_latents
		from utils.image import decode_latent
		from video import save_frames_to_png, save_video
		from utils import prompt_to_seed
		
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
		
		frames = interpolate_latents(
			latents,
			kwargs.get("num_interpolation_steps", 10),
			kwargs.get("looping_mode", False),
			kwargs.get("circular_motion", False),
			kwargs.get("tile_rows", 1),
			kwargs.get("tile_cols", 1),
			kwargs.get("quiet", False),
			patchwise=kwargs.get("patchwise_morph", False),
			patch_grid=kwargs.get("patch_grid", 4),
			patch_ratio=kwargs.get("patch_ratio", 0.3),
			patch_shuffle_mode=kwargs.get("patch_shuffle_mode", "none"),
			patch_strength_mode=kwargs.get("patch_strength_mode", "uniform"),
			patch_mask_type=kwargs.get("patch_mask_type", "sin"),
			patch_mask_image=kwargs.get("patch_mask_image", "")

		)
		
		images = [decode_latent(self.pipe, latent, show=kwargs.get("show_preview")) for latent in frames]
		save_frames_to_png(images, out_dir=os.path.join(kwargs["output_dir"], "frames"))
		video_path = save_video(images, width, height, kwargs["frames_per_second"], os.path.join(kwargs["output_dir"], "output"))
		yield video_path


	def predict_infinite(self, **kwargs) -> Iterator[np.ndarray]:
		from generation.local import generate_from_local
		from generation.replicate import generate_from_replicate
		from generation.interpolate import interpolate_latents
		from prompts.chainer import chain_prompt_ai
		# from utils.image import decode_latent
		from utils import prompt_to_seed
		
		prompt = kwargs.get("prompt_start", "A beautiful abstract pattern")
		output_dir = kwargs.get("output_dir", "output")
		os.makedirs(output_dir, exist_ok=True)
		prompt_log_path = os.path.join(output_dir, "prompt_log.txt")
		
		# Clear prompt log
		with open(prompt_log_path, "w") as f:
			f.write("")
		
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
			frames = interpolate_latents(
				[prev_latent, new_latent],
				steps,
				looping=False,
				circular=False,
				tile_rows=1,
				tile_cols=1,
				quiet=kwargs.get("quiet", False),
				patchwise=kwargs.get("patchwise_morph", False),
				patch_grid=kwargs.get("patch_grid", 4),
				patch_ratio=kwargs.get("patch_ratio", 0.3),
				patch_shuffle_mode=kwargs.get("patch_shuffle_mode", "none"),
				patch_strength_mode=kwargs.get("patch_strength_mode", "uniform"),
				patch_mask_animated=kwargs.get("patch_mask_animated", False),
				patch_mask_speed=kwargs.get("patch_mask_speed", 0.5),
				patch_mask_threshold=kwargs.get("patch_mask_threshold", 0.4)

			)
		
			for frame in frames:
				image = decode_latent(self.pipe, frame)
				frame_path = os.path.join(output_dir, f"frame_{frame_counter:06d}.png")
				Image.fromarray(image).save(frame_path)
				frame_counter += 1
				yield image
		
			prev_latent = new_latent
			prev_prompt = new_prompt

