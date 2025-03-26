import argparse
from predictor import Predictor

def main():
	parser = argparse.ArgumentParser(description="🛰️ Cosmist CLI Generator")

	# Prompts
	parser.add_argument("--prompt_start", type=str, default="", help="Start prompt")
	parser.add_argument("--prompt_end", type=str, default="", help="End prompt (finite)")
	parser.add_argument("--prompt_list_path", type=str, default="", help="Prompt list file path")
	parser.add_argument("--theme", type=str, default="cosmist", help="Prompt theme")

	# Output resolution
	parser.add_argument("--resolution", type=str, default="512x512", help="Image resolution")

	# Generation details
	parser.add_argument("--num_keyframes", type=int, default=8)
	parser.add_argument("--num_output_frames", type=int, default=None)
	parser.add_argument("--num_interpolation_steps", type=int, default=10)
	parser.add_argument("--num_inference_steps", type=int, default=50)
	parser.add_argument("--guidance_start", type=float, default=7.5)
	parser.add_argument("--guidance_end", type=float, default=7.5)
	parser.add_argument("--noise_scale", type=float, default=1.0)
	parser.add_argument("--master_seed", type=int, default=None)
	parser.add_argument("--transition_duration", type=float, default=None, help="Duration per morph (s)")

	# Animation + layout
	parser.add_argument("--tile_rows", type=int, default=1)
	parser.add_argument("--tile_cols", type=int, default=1)
	parser.add_argument("--looping_mode", action="store_true")
	parser.add_argument("--loop_padding", type=int, default=0)
	parser.add_argument("--circular_motion", action="store_true")

	# Rendering & output
	parser.add_argument("--frames_per_second", type=int, default=24)
	# parser.add_argument("--preview_fps", type=int, default=20)
	parser.add_argument("--save_frames", action="store_true")
	# parser.add_argument("--show_preview", action="store_true")
	parser.add_argument("--output_dir", type=str, default="./output")
	parser.add_argument("--tileable", action="store_true")

	# Mode
	parser.add_argument("--infinite_mode", action="store_true")
	parser.add_argument("--ai_chain_prompts", action="store_true")

	# Backend: replicate
	parser.add_argument("--use_replicate", action="store_true")
	parser.add_argument("--replicate_model", type=str, default="tstramer/material-diffusion:a42692c54c0f407f803a0a8a9066160976baedb77c91171a01730f9b0d7beeff")
	parser.add_argument("--replicate_token", type=str, default="")
	parser.add_argument("--patchwise_morph", action="store_true", help="Interpolate patches of latent instead of full image")
	parser.add_argument("--patch_grid", type=int, default=4, help="Number of patches per side for patchwise morph")
	parser.add_argument("--patch_ratio", type=float, default=0.3, help="Fraction of patches to morph per frame")
	parser.add_argument("--patch_shuffle_mode", type=str, default="none",
	choices=["none", "per_transition", "per_frame"],
	help="Shuffle patch order randomly to create glitchy motion")
	parser.add_argument("--patch_motion", type=str, default="none",
	choices=["none", "drift"],
	help="Apply motion to patches during morph (e.g. drifting motion trails)")
	parser.add_argument("--patch_strength_mode", type=str, default="uniform",
	choices=["uniform", "radial", "random", "stripe"],
	help="How to vary morph strength across patches")
	parser.add_argument("--patch_mask_animated", action="store_true",
		help="Animate patch mask over time for smooth evolving regions")
	
	parser.add_argument("--patch_mask_speed", type=float, default=0.5,
		help="Speed of animated patch mask")
	
	parser.add_argument("--patch_mask_threshold", type=float, default=0.4,
		help="Threshold for activating a patch in animated mask (0–1)")


	# Parse args
	args = parser.parse_args()
	args_dict = vars(args)

	# Instantiate and configure predictor
	predictor = Predictor(replicate_model=args_dict["replicate_model"])
	predictor.tileable = args_dict.pop("tileable", False)
	predictor.setup()

	try:
		for _ in predictor.predict(**args_dict):
			pass
	except KeyboardInterrupt:
		print("🛑 Interrupted. Exiting.")

if __name__ == "__main__":
	main()
