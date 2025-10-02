# Instructions for RGB-D Image Segmentation

This document provides instructions for setting up the environment, preparing the dataset, and running the training and testing scripts for the RGB-D image segmentation model.

## 1. Environment Setup

It is recommended to use a virtual environment to manage dependencies.

### Using Conda
```bash
# Create and activate a new conda environment
conda create -n rgdb-seg python=3.8
conda activate rgdb-seg
```

### Using venv
```bash
# Create and activate a new virtual environment
python3 -m venv rgdb-seg-env
source rgdb-seg-env/bin/activate
```

## 2. Install Dependencies

Install the required Python libraries using pip:
```bash
pip install torch torchvision
pip install opencv-python scikit-image pandas timm matplotlib
```

## 3. Pre-trained Checkpoint

The model uses a pre-trained Vision Transformer (ViT) from the Segment Anything Model (SAM).

1.  **Download the checkpoint:**
    You can download the pre-trained `vit_b` checkpoint from the official source:
    [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

2.  **Place the checkpoint:**
    Create a directory named `pretrained_checkpoint` in the root of the project and place the downloaded file there:
    ```
    project_root/
    ├── pretrained_checkpoint/
    │   └── vit_b.pth
    ├── ...
    ```

## 4. Dataset Preparation

This project is configured to use the **NEU-RSDDS-AUG** dataset. The expected directory structure is as follows:
```
datasets/
└── NEU-RSDDS-AUG/
    ├── Image_train/       # BMP format
    ├── Depth_train/       # TIFF format
    ├── GT_train/          # PNG format
    ├── Image_test/        # BMP format
    └── Depth_test/        # TIFF format
```
Place the dataset in a directory named `datasets` at the same level as the project root.

## 5. Training

To train the model, run the `main.py` script. The script will automatically use the training data from the `../datasets/NEU-RSDDS-AUG` directory.

```bash
python main.py
```

- **Checkpoints** will be saved to `/hy-tmp/output/checkpoint.pth`.
- **Logs** will be saved to `/hy-tmp/output/result.log`.

If you are running in a local environment without the `/hy-tmp` directory, the script will use `./hy-tmp` instead.

## 6. Testing

To test the model, you need a trained checkpoint.

1.  **Run the evaluation script:**
    Use the `--eval` flag and provide the path to your trained checkpoint with the `--restore-model` argument.

    ```bash
    python main.py --eval --restore-model ./hy-tmp/output/checkpoint.pth
    ```

2.  **Prediction Masks:**
    The predicted masks will be saved in PNG format in the `/hy-tmp/output/predictions/` directory, with each mask resized to its original image dimensions.

    As with training, if the `/hy-tmp` directory is not available, the script will use `./hy-tmp` for output.