# prompts/chainer.py

import os
import openai

def chain_prompt_ai(prev_prompt: str, counter: int) -> str:
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
			{"role": "user", "content": prompt_message}
		],
		max_tokens=50,
		temperature=0.7,
	)
	return response["choices"][0]["message"]["content"].strip()
