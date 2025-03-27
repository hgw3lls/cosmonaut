import os
import requests
from io import BytesIO
from datetime import datetime
from typing import Iterator, Any

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import openai
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from dotenv import load_dotenv

from utils import prompt_to_seed, slerp
from video import save_video, save_frames_to_png, save_metadata
from config import RESOLUTION_MAP
from interpolate import interpolate_latents

load_dotenv()
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")


class Predictor:
	def __init__(
		self,
		replicate_model: str = "tstramer/material-diffusion:a42692c54c0f407f803a0a8a9066160976baedb77c91171a01730f9b0d7beeff",
	):
		self.tileable = False
		self.replicate_model = replicate_model
		self.pipe = None

	def setup(self) -> None:
		"""Load the local diffusion model."""
		print("🛰️ Loading local diffusion model...")
		self.pipe = DiffusionPipeline.from_pretrained(
			"runwayml/stable-diffusion-v1-5",
			torch_dtype=torch.float32,
			cache_dir="diffusers-cache",
		)
		self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
			self.pipe.scheduler.config
		)
		self.pipe.to("cpu")

	def predict(self, **kwargs) -> Iterator[Any]:
		"""
		Main prediction entry point. Chooses infinite or finite prediction
		based on the `infinite_mode` flag.
		"""
		if kwargs.get("transition_duration") and not kwargs.get("num_interpolation_steps"):
			fps = kwargs.get("preview_fps") or kwargs.get("frames_per_second") or 20
			kwargs["num_interpolation_steps"] = int(fps * kwargs["transition_duration"])
			print(
				f"🌀 Using transition_duration={kwargs['transition_duration']}s → "
				f"{kwargs['num_interpolation_steps']} interpolation steps at {fps} FPS"
			)
		if kwargs.get("infinite_mode"):
			return self.predict_infinite(**kwargs)
		else:
			return self.predict_finite(**kwargs)

	def predict_infinite(self, **kwargs) -> Iterator[np.ndarray]:
		"""
		Generate an infinite sequence of images by continuously refining prompts.
		"""
		prompt_start = kwargs.get("prompt_start", "Abstract tessellated texture")
		output_dir = kwargs.get("output_dir", "output")
		os.makedirs(output_dir, exist_ok=True)
		preview_dir = os.path.join(output_dir, "frames")
		os.makedirs(preview_dir, exist_ok=True)
		prompt_log_path = os.path.join(output_dir, "prompt_log.txt")

		frame_counter = 0
		prev_prompt = prompt_start
		if kwargs.get("resume") and os.path.exists(prompt_log_path):
			with open(prompt_log_path, "r") as f:
				lines = f.readlines()
				if lines:
					last_line = lines[-1].strip()
					if "] " in last_line:
						frame_counter = int(last_line.split("]")[0][1:])
						prev_prompt = last_line.split("] ")[-1]
		else:
			with open(prompt_log_path, "w") as f:
				f.write("")

		# Generate initial latent
		seed = prompt_to_seed(prev_prompt)
		use_replicate = kwargs.get("use_replicate", False)
		num_inference_steps = kwargs["num_inference_steps"]
		noise_scale = kwargs.get("noise_scale")
		prev_latent = (
			self.generate_from_replicate(prev_prompt, 512, 512, num_inference_steps, self.replicate_model)
			if use_replicate
			else self.generate_from_local(prev_prompt, seed, 512, 512, num_inference_steps, noise_scale, False)
		)

		while True:
			# Chain prompt if desired
			if kwargs.get("ai_chain_prompts"):
				new_prompt = self.chain_prompt_ai(prev_prompt, frame_counter)
			else:
				new_prompt = f"{prev_prompt} {frame_counter}"

			with open(prompt_log_path, "a") as f:
				f.write(f"[{frame_counter:06d}] {new_prompt}\n")

			seed = prompt_to_seed(new_prompt)
			new_latent = (
				self.generate_from_replicate(new_prompt, 512, 512, num_inference_steps, self.replicate_model)
				if use_replicate
				else self.generate_from_local(new_prompt, seed, 512, 512, num_inference_steps, noise_scale, False)
			)

			# Interpolate between previous and new latent
			frames = interpolate_latents(
				[prev_latent, new_latent],
				steps=kwargs.get("num_interpolation_steps", 10),
				looping=False,
				circular=False,
				tile_rows=1,
				tile_cols=1,
				quiet=False,
				patchwise=kwargs.get("patchwise_morph", False),
				patch_grid=kwargs.get("patch_grid", 4),
				patch_ratio=kwargs.get("patch_ratio", 0.3),
				patch_shuffle_mode=kwargs.get("patch_shuffle_mode", "none"),
				patch_motion=kwargs.get("patch_motion", "none"),
				patch_strength_mode=kwargs.get("patch_strength_mode", "uniform"),
				patch_mask_animated=kwargs.get("patch_mask_animated", False),
				patch_mask_speed=kwargs.get("patch_mask_speed", 0.5),
				patch_mask_threshold=kwargs.get("patch_mask_threshold", 0.4),
				patch_mask_type=kwargs.get("patch_mask_type", "sin"),
				patch_mask_blend=kwargs.get("patch_mask_blend", ""),
				patch_mask_blend_mode=kwargs.get("patch_mask_blend_mode", "multiply"),
				patch_mask_image=kwargs.get("patch_mask_image", "")
			)

			for frame in frames:
				image = self.decode_latent(frame)
				frame_path = os.path.join(preview_dir, f"frame_{frame_counter:06d}.png")
				Image.fromarray(image).save(frame_path)
				frame_counter += 1
				yield image

			prev_latent = new_latent
			prev_prompt = new_prompt

	def decode_latent(self, latent: Any, show: bool = False) -> np.ndarray:
		"""Decode a latent representation into an image array."""
		with torch.no_grad():
			if isinstance(latent, np.ndarray):
				latent = torch.from_numpy(latent).to(torch.float32)
			elif isinstance(latent, torch.Tensor):
				latent = latent.to(torch.float32)
			decoded = self.pipe.decode_latents(latent)
			if isinstance(decoded, torch.Tensor):
				decoded = decoded.detach().cpu().permute(0, 2, 3, 1).numpy()
			image = (decoded[0] * 255).round().astype(np.uint8)
			return image

	def generate_from_replicate(
		self, prompt: str, width: int, height: int, steps: int, model_name: str
	) -> torch.Tensor:
		"""Generate latent using the Replicate API."""
		import replicate  # local import
		output = replicate.run(
			model_name,
			input={
				"prompt": prompt,
				"width": width,
				"height": height,
				"cfg": 3.5,
				"steps": steps,
				"prompt_strength": 0.85,
			},
		)
		response = requests.get(output[0])
		image = Image.open(BytesIO(response.content)).convert("RGB")
		image = image.resize((width, height))
		image_np = np.array(image).astype(np.float32) / 255.0
		image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
		image_tensor = image_tensor * 2 - 1
		with torch.no_grad():
			latent_dist = self.pipe.vae.encode(image_tensor.to(self.pipe.device)).latent_dist
			latent = latent_dist.sample() * 0.18215
		return latent

	def generate_from_local(
		self, prompt: str, seed: int, width: int, height: int, steps: int, noise_scale: float, quiet: bool
	) -> torch.Tensor:
		"""Generate latent using the local diffusion model."""
		generator = torch.Generator("cpu").manual_seed(seed)
		noise = torch.randn(
			(1, self.pipe.unet.in_channels, height // 8, width // 8), generator=generator
		) * noise_scale
		text_emb = self.pipe._encode_prompt(prompt, "cpu", 1, True, "")
		self.pipe.scheduler.set_timesteps(steps)
		return self.denoise(noise, text_emb, steps, 7.5, generator, quiet=quiet)

	def denoise(
		self, latents: torch.Tensor, text_embeddings: torch.Tensor, steps: int, guidance_scale: float, generator: torch.Generator, quiet: bool = False
	) -> torch.Tensor:
		"""Perform the denoising process."""
		scheduler = self.pipe.scheduler
		extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta=0)
		for t in tqdm(scheduler.timesteps, desc="Denoising", disable=quiet):
			latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
			latent_input = scheduler.scale_model_input(latent_input, t)
			noise_pred = self.pipe.unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
			if guidance_scale > 1.0:
				uncond, cond = noise_pred.chunk(2)
				noise_pred = uncond + guidance_scale * (cond - uncond)
			latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
		return latents

	def chain_prompt_ai(self, prev_prompt: str, counter: int) -> str:
		"""
		Use OpenAI's API to generate a new creative prompt from the previous one.
		"""
		openai.api_key = os.getenv("OPENAI_API_KEY")
		if not openai.api_key:
			raise ValueError("OPENAI_API_KEY not set")
		prompt_message = (
			f"Refine this prompt into a new creative abstract infinite pattern:\n\n"
			f"'{prev_prompt}'\n\nNew version:"
		)
		response = openai.ChatCompletion.create(
			model="gpt-3.5-turbo",
			messages=[
				{"role": "system", "content": "You are a prompt chaining assistant for generative art."},
				{"role": "user", "content": prompt_message},
			],
			max_tokens=50,
			temperature=0.7,
		)
		return response["choices"][0]["message"]["content"].strip()

	def predict_finite(self, **kwargs) -> Iterator[str]:
		"""
		Generate a finite sequence of images from a list of prompts.
		"""
		prompts = kwargs.get("prompts", [])
		resolution = kwargs.get("resolution", "512x512")
		if resolution not in RESOLUTION_MAP:
			raise ValueError(f"Unsupported resolution '{resolution}'")
		width, height = RESOLUTION_MAP[resolution]
		output_dir = kwargs.get("output_dir", "output")
		os.makedirs(output_dir, exist_ok=True)

		seeds = [prompt_to_seed(p) for p in prompts]
		latents = []
		num_inference_steps = kwargs["num_inference_steps"]
		noise_scale = kwargs.get("noise_scale")
		use_replicate = kwargs.get("use_replicate", False)
		quiet = kwargs.get("quiet", False)

		for i, prompt in enumerate(prompts):
			latent = (
				self.generate_from_replicate(prompt, width, height, num_inference_steps, self.replicate_model)
				if use_replicate
				else self.generate_from_local(prompt, seeds[i], width, height, num_inference_steps, noise_scale, quiet)
			)
			latents.append(latent)

		frames = interpolate_latents(
			latents,
			steps=kwargs.get("num_interpolation_steps", 10),
			looping=kwargs.get("looping_mode", False),
			circular=kwargs.get("circular_motion", False),
			tile_rows=kwargs.get("tile_rows", 1),
			tile_cols=kwargs.get("tile_cols", 1),
			quiet=quiet,
			patchwise=kwargs.get("patchwise_morph", False),
			patch_grid=kwargs.get("patch_grid", 4),
			patch_ratio=kwargs.get("patch_ratio", 0.3),
			patch_shuffle_mode=kwargs.get("patch_shuffle_mode", "none"),
			patch_motion=kwargs.get("patch_motion", "none"),
			patch_strength_mode=kwargs.get("patch_strength_mode", "uniform"),
			patch_mask_animated=kwargs.get("patch_mask_animated", False),
			patch_mask_speed=kwargs.get("patch_mask_speed", 0.5),
			patch_mask_threshold=kwargs.get("patch_mask_threshold", 0.4),
			patch_mask_type=kwargs.get("patch_mask_type", "sin"),
			patch_mask_blend=kwargs.get("patch_mask_blend", ""),
			patch_mask_blend_mode=kwargs.get("patch_mask_blend_mode", "multiply"),
			patch_mask_image=kwargs.get("patch_mask_image", "")
		)

		images = [self.decode_latent(latent) for latent in frames]

		if kwargs.get("save_frames"):
			frames_dir = os.path.join(output_dir, "frames")
			os.makedirs(frames_dir, exist_ok=True)
			save_frames_to_png(images, out_dir=frames_dir)

		video_path = None
		if not kwargs.get("no_video"):
			basename = os.path.join(output_dir, f"output_{datetime.now().strftime('%Y-%m-%d_%H-%M')}")
			video_path = save_video(images, width, height, kwargs["frames_per_second"], basename)

		if video_path:
			yield video_path
