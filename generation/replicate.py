# generation/replicate.py

import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO

def generate_from_replicate(pipe, prompt, width, height, steps, model_name):
	import replicate
	output = replicate.run(model_name, input={
		"prompt": prompt,
		"width": width,
		"height": height,
		"cfg": 3.5,
		"steps": steps,
		"prompt_strength": 0.85
	})
	response = requests.get(output[0])
	image = Image.open(BytesIO(response.content)).convert("RGB")
	image = image.resize((width, height))
	image_np = np.array(image).astype(np.float32) / 255.0
	image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
	image_tensor = image_tensor * 2 - 1
	with torch.no_grad():
		latent_dist = pipe.vae.encode(image_tensor.to(pipe.device)).latent_dist
		latent = latent_dist.sample() * 0.18215
	return latent
