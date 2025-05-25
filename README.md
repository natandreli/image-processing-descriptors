# Image Descriptor Project

This project uses the [Oxford 102 Flowers Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) to extract image descriptors (like HOG) and train a classifier.

## 📁 Folder Structure

After setup, your project should look like this:

```plaintext
project-root/
├── main.py
├── prepare_dataset.py
├── data/
│   ├── flowers/               # Contains all the JPG images
│   │   ├── image_00001.jpg
│   │   └── ...
│   ├── labels/
│   |   └── imagelabels.mat    # MATLAB file with labels (1 to 102)
|   ├── dataset/               # This will be generated automatically
```

## ⬇️ 1. Download the Dataset

From the official site:

- [images (.tgz)](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
- [labels (.mat)](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat)

Steps:

1. Unpack `102flowers.tgz`
2. Move all the `.jpg` images to:
```plaintext
data/flowers/
```
3. Move the `imagelabels.mat` file to:
```plaintext
data/labels/
```
Create those folders manually if needed.

## ⚙️ 2. Organize the Images by Class
Run the following script once to organize the images by class using their labels:
```bash
python prepare_dataset.py
```
This will generate:
```plaintext
data/dataset/class_001/
data/dataset/class_002/
...
data/dataset/class_102/
```
Each folder contains images for one specific flower class.

## ✅ Next Steps
After the dataset is prepared, you're ready to run the main pipeline:
```bash
python main.py
```
That will:
- Load and resize all images
- Extract HOG descriptors
- Train a classifier (SVM)
- Show evaluation results
