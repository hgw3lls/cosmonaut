import os
import tkinter as tk
from PIL import Image, ImageTk
import argparse

class InfiniteTileApp:
	def __init__(self, folder, fps=24, prompt="N/A", tile=True, grid_rows=2, grid_cols=2):
		self.folder = folder
		self.fps = fps
		self.delay = int(1000 / fps)
		self.prompt = prompt
		self.tile = tile
		self.grid_rows = grid_rows
		self.grid_cols = grid_cols
		self.hud_visible = False
		self.paused = False  # starts playing
		self.index = 0

		# Build a sorted list of PNG images.
		self.images = sorted(
			[os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.png')]
		)
		if not self.images:
			print("No PNG images found in the specified folder.")
			exit(1)

		self.root = tk.Tk()
		self.root.title("Infinite Tile Preview")
		# Load first image to set window size.
		first_img = Image.open(self.images[0])
		self.root.geometry(f"{first_img.width}x{first_img.height}")

		self.label = tk.Label(self.root)
		self.label.pack(expand=True, fill="both")

		# HUD label (initially hidden)
		self.hud_label = tk.Label(self.root, bg="black", fg="white", font=("Helvetica", 12), justify="left")
		self.hud_label.place_forget()

		# Bind keys:
		self.root.bind("h", self.toggle_hud)        # Toggle HUD.
		self.root.bind("<space>", self.toggle_pause) # Pause/resume.
		self.root.bind("t", self.toggle_tiling)       # Toggle tiling mode.
		self.root.bind("<Up>", self.increase_grid_rows)    # Increase grid rows.
		self.root.bind("<Down>", self.decrease_grid_rows)  # Decrease grid rows.
		self.root.bind("<Right>", self.increase_grid_cols)  # Increase grid columns.
		self.root.bind("<Left>", self.decrease_grid_cols)   # Decrease grid columns.
		self.root.bind("+", self.increase_fps)         # Increase FPS.
		self.root.bind("-", self.decrease_fps)         # Decrease FPS.

		self.update_image()
		self.root.mainloop()

	def update_image(self):
		if not self.paused:
			current_path = self.images[self.index]
			try:
				img = Image.open(current_path)
			except Exception as e:
				print("Error opening image:", e)
				self.index = (self.index + 1) % len(self.images)
				self.root.after(self.delay, self.update_image)
				return

			# Get current window size.
			win_w = max(1, self.root.winfo_width())
			win_h = max(1, self.root.winfo_height())
			
			if self.tile:
				# Calculate cell size exactly.
				cell_w = win_w // self.grid_cols
				cell_h = win_h // self.grid_rows
				# Force resize to fill the cell exactly.
				img_cell = img.resize((cell_w, cell_h), Image.ANTIALIAS)
				# Create composite image with no gaps.
				composite = Image.new("RGB", (cell_w * self.grid_cols, cell_h * self.grid_rows))
				for r in range(self.grid_rows):
					for c in range(self.grid_cols):
						composite.paste(img_cell, (c * cell_w, r * cell_h))
				final_img = composite
			else:
				# If not tiling, scale image to fit within window (with margin).
				margin = 100
				target_w = max(1, win_w - margin)
				target_h = max(1, win_h - margin)
				img.thumbnail((target_w, target_h), Image.ANTIALIAS)
				final_img = img

			tk_img = ImageTk.PhotoImage(final_img)
			self.label.config(image=tk_img)
			self.label.image = tk_img  # Prevent garbage collection.
			self.index = (self.index + 1) % len(self.images)

		# Update HUD if visible.
		if self.hud_visible:
			state = "Paused" if self.paused else "Playing"
			current_image = self.images[self.index - 1] if self.index > 0 else self.images[0]
			hud_text = (
				f"State: {state}\n"
				f"FPS: {self.fps}\n"
				f"Image: {os.path.basename(current_image)}\n"
				f"Prompt: {self.prompt}\n"
				f"Tiling: {'On' if self.tile else 'Off'}\n"
				f"Grid: {self.grid_rows} x {self.grid_cols}"
			)
			self.hud_label.config(text=hud_text)
			self.hud_label.lift()

		self.root.after(self.delay, self.update_image)

	def toggle_pause(self, event=None):
		self.paused = not self.paused
		print("Paused" if self.paused else "Playing")

	def toggle_tiling(self, event=None):
		self.tile = not self.tile
		print("Tiling toggled", "On" if self.tile else "Off")

	def increase_grid_rows(self, event=None):
		self.grid_rows += 1
		print("Grid rows increased to", self.grid_rows)

	def decrease_grid_rows(self, event=None):
		if self.grid_rows > 1:
			self.grid_rows -= 1
			print("Grid rows decreased to", self.grid_rows)
		else:
			print("Grid rows already at minimum (1)")

	def increase_grid_cols(self, event=None):
		self.grid_cols += 1
		print("Grid columns increased to", self.grid_cols)

	def decrease_grid_cols(self, event=None):
		if self.grid_cols > 1:
			self.grid_cols -= 1
			print("Grid columns decreased to", self.grid_cols)
		else:
			print("Grid columns already at minimum (1)")

	def toggle_hud(self, event=None):
		self.hud_visible = not self.hud_visible
		if self.hud_visible:
			self.hud_label.place(x=10, y=10)
		else:
			self.hud_label.place_forget()

	def increase_fps(self, event=None):
		self.fps += 1
		self.delay = int(1000 / self.fps)
		print("FPS increased to", self.fps)

	def decrease_fps(self, event=None):
		if self.fps > 1:
			self.fps -= 1
			self.delay = int(1000 / self.fps)
			print("FPS decreased to", self.fps)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Infinite scaling tile preview with adjustable grid.")
	parser.add_argument("--folder", required=True, help="Folder with PNG images")
	parser.add_argument("--fps", type=int, default=24, help="Frames per second")
	parser.add_argument("--prompt", type=str, default="N/A", help="Prompt text for HUD")
	parser.add_argument("--tile", action="store_true", help="Start in tiling mode")
	parser.add_argument("--grid_rows", type=int, default=2, help="Initial number of grid rows")
	parser.add_argument("--grid_cols", type=int, default=2, help="Initial number of grid columns")
	args = parser.parse_args()

	InfiniteTileApp(
		folder=args.folder,
		fps=args.fps,
		prompt=args.prompt,
		tile=args.tile,
		grid_rows=args.grid_rows,
		grid_cols=args.grid_cols
	)
