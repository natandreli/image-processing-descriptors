import os
import shutil
from scipy.io import loadmat

base_path = os.path.dirname(__file__)
images_path = os.path.join(base_path, "data", "flowers") 
labels_path = os.path.join(base_path, "data", "labels", "imagelabels.mat")
output_path = os.path.join(base_path, "data", "dataset")

labels = loadmat(labels_path)["labels"][0] 

os.makedirs(output_path, exist_ok=True)
for i in range(1, 103):
    os.makedirs(os.path.join(output_path, f"class_{i:03d}"), exist_ok=True)

for idx, label in enumerate(labels):
    img_name = f"image_{idx+1:05d}.jpg"
    src = os.path.join(images_path, img_name)
    dst = os.path.join(output_path, f"class_{label:03d}", img_name)

    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"[WARNING] Image not found: {src}")

print("✅ Dataset reorganized by class in: data/dataset/")
