import os
from pathlib import Path
import logging

#  Logging stream

logging.basicConfig(level=logging.INFO,format='[%(ascitime)s]: %(message)s')

files = [
    "src/__init__.py",
    "src/helpers.py",
    ".env",
    "setup.py",
    "app.py",
    "/data",
    "notebooks/notebook.ipynb",
    "templates"
]


for filepath in files:
    filepath = Path(filepath)
    filedir,filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filename) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")