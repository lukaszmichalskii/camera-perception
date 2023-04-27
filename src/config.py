import os
import pathlib

ROOT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent
DETECTOR = ROOT_DIR.joinpath("docs/yolo")
