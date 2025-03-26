import os
import time
import threading
import argparse
from PIL import Image, ImageTk
import tkinter as tk

class LivePreviewApp:
	def __init__(self, folder, fps=24):
		self.folder = folder
		self.delay = int(1000 / fps)
		self.frame_paths = []
		self.seen = set()
		self.current_index = 0
		self.running = True
		self.paused = False

		self.root = tk.Tk()
		self.root.title("🌀 Cosmist Live Preview")
		self.label = tk.Label(self.root)
		self.label.pack()

		self.root.bind("<space>", self.toggle_pause)
		self.root.bind("+", self.increase_fps)
		self.root.bind("-", self.decrease_fps)
		self.root.bind("q", self.quit_app)

		threading.Thread(target=self.refresh_loop, daemon=True).start()
		self.root.after(self.delay, self.update_frame)
		self.root.mainloop()

	def refresh_loop(self):
		while self.running:
			all_pngs = sorted([
				os.path.join(self.folder, f) for f in os.listdir(self.folder)
				if f.endswith(".png")
			])
			new = [f for f in all_pngs if f not in self.seen]
			self.frame_paths.extend(new)
			self.seen.update(new)
			time.sleep(1)

	def update_frame(self):
		if self.running and not self.paused and self.frame_paths:
			frame_path = self.frame_paths[self.current_index % len(self.frame_paths)]
			try:
				img = Image.open(frame_path)
				img = self.scale_image(img)
				tk_img = ImageTk.PhotoImage(img)
				self.label.configure(image=tk_img)
				self.label.image = tk_img
			except Exception as e:
				print(f"⚠️ Failed to load image: {e}")
			self.current_index += 1
		self.root.after(self.delay, self.update_frame)

	def scale_image(self, img):
		screen_w = self.root.winfo_screenwidth()
		screen_h = self.root.winfo_screenheight()
		img.thumbnail((screen_w - 100, screen_h - 100), Image.ANTIALIAS)
		return img

	def toggle_pause(self, event=None):
		self.paused = not self.paused
		print("⏸️ Paused" if self.paused else "▶️ Resumed")

	def increase_fps(self, event=None):
		self.delay = max(10, self.delay - 10)
		print(f"⚡ Increased FPS: {1000 // self.delay}")

	def decrease_fps(self, event=None):
		self.delay = min(1000, self.delay + 10)
		print(f"🐢 Decreased FPS: {1000 // self.delay}")

	def quit_app(self, event=None):
		print("👋 Exiting preview.")
		self.running = False
		self.root.destroy()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="🎞️ Cosmist Real-Time Preview GUI")
	parser.add_argument("--folder", type=str, required=True, help="Folder with .png frames")
	parser.add_argument("--fps", type=int, default=24, help="Initial frames per second")
	args = parser.parse_args()

	LivePreviewApp(args.folder, fps=args.fps)
