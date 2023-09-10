
"""
This code is used to create the project folder structure template and required file for the project
"""

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

project_name = "mlproject"

list_of_files= [
    ".github/workflows/.gitkeep",
    f"src/__init__.py",
    f"src/components/__init__.py",
    f"src/pipeline/__init__.py",
    f"src/exception.py",
    f"src/logger.py",
    f"src/utils.py",
    f"templates/__init__.py",
    f"templates/home.html",
    f"templates/index.html",
    "app.py",
    "requirements.txt",
    "setup.py",
    ".gitignore",

]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"creating a directory:{filedir} for the file name {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"creating empty file {filepath}")

    else:
        logging.info(f'{filename} is already exists')









