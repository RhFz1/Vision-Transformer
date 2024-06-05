import os
import zipfile
import requests
from pathlib import Path

# Data download path, credits: mrdbourke
data_down_url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"


data_path = Path("/home/syednoor/Desktop/SAM/data")
image_path = data_path / "images"


# Code for creating the directories where the raw images shall be stored.
if image_path.is_dir():
    print (f"{image_path} directory already exists!!")
else:
    print(f"Creating {image_path} directory!!")
    image_path.mkdir(parents=True, exist_ok=True, mode=0o770)

# Downloading the files
with open(data_path / "images.zip", mode="wb") as file:
    request = requests.get(data_down_url)
    print("Downloading the data")
    file.write(request.content)

# Unzipping the files
with zipfile.ZipFile(data_path / "images.zip", 'r') as zipref:
    print("Unzipping the file contents")
    zipref.extractall(image_path)

os.remove(data_path / "images.zip")
