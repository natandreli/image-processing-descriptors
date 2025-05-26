# Image Descriptor Project

This project uses the [Flowers Dataset](https://public.roboflow.com/classification/flowers) to extract image descriptors (like HOG and LAB) and train a classifier.

## 📁 Folder Structure

After setup, your project should look like this:

```plaintext
project-root/
├── main.py
├── test/
|   ├── daisy/
|   ├── dandelion/
├── data/
|   ├── dataset/
|   |   ├── daisy
|   |   ├── dandelion
```

## ⬇️ 1. Download the Dataset

Download the dataset from Google Drive. A subset of images was excluded from the training process to be used later for testing the model's performance.

- [dataset](https://drive.google.com/file/d/1Ymrb3BJ1SKbQyozEI0vxkBMR69uUTN-k/view?usp=sharing)

Steps:

1. Unpack `data.zip`
2. Move the data folder to the root of the project

Each folder contains images for one specific flower class.

## ✅ Next Steps

After the dataset is prepared, you're ready to run the main pipeline:

```bash
python main.py
```

That will:

- Load and resize all images
- Extract HOG and LAB descriptors
- Train a classifier (SVM)
- Show evaluation results

Run the command:

```bash
python gui_app.py
```

A GUI will open where you can upload images to be classified.
In the test folder there are some images to test the model.
