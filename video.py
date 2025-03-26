import os
import av
import cv2
from datetime import datetime
from PIL import Image

def save_frames_to_png(images, out_dir="frames"):
	os.makedirs(out_dir, exist_ok=True)
	for idx, img in enumerate(images):
		Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).save(
			os.path.join(out_dir, f"frame_{idx:04d}.png")
		)

def save_video(images, width, height, fps, basename):
	output_path = f"{basename}.mp4"
	container = av.open(output_path, mode="w")
	stream = container.add_stream("h264", rate=fps)
	stream.width = width
	stream.height = height
	stream.pix_fmt = "yuv420p"

	for img in images:
		frame = av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), format="bgr24")
		packet = stream.encode(frame)
		container.mux(packet)
	container.mux(stream.encode())
	container.close()
	return output_path

def save_metadata(basename, prompts, theme, resolution, looping_mode):
	with open(f"{basename}_prompts.txt", "w") as f:
		f.write(f"Theme: {theme}\nResolution: {resolution}\nLooping: {looping_mode}\n")
		for i, p in enumerate(prompts):
			f.write(f"Prompt {i+1}: {p}\n")
