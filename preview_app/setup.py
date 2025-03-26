from setuptools import setup

APP = ['preview_loop.py']
DATA_FILES = []
OPTIONS = {
	'argv_emulation': True,
	'packages': ['PIL', 'tkinter'],
	'iconfile': None  # Add .icns icon if you want
}

setup(
	app=APP,
	name='Cosmist Live Preview',
	data_files=DATA_FILES,
	options={'py2app': OPTIONS},
	setup_requires=['py2app'],
)