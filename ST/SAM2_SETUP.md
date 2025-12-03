# SAM2 Setup Guide

## 1. Installation

To use SAM2 for tadpole segmentation, you need to install the `sam2` Python package.
The official repository is: https://github.com/facebookresearch/sam2

**Command:**
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

*Note: This requires `torch>=2.5.1` and `torchvision>=0.20.1`.*

## 2. Checkpoint Download

You need the `sam2_hiera_base.pt` (or `sam2.1_hiera_base_plus.pt` depending on version, but the code defaults to `sam2_hiera_base.pt` for the older release or you can use the new 2.1 versions if config is updated).

**Download Link (Official):**
- [sam2_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)

*Recommendation:* The code looks for `sam2_hiera_base.pt` by default. You can rename the downloaded file or update the environment variable.

**Placement:**
Place the `.pt` file in the `checkpoints/` folder in the project root, or anywhere you like.

## 3. Environment Configuration

Set the environment variable `SAM2_CHECKPOINT` to the full path of the `.pt` file.

**Example (Linux/Mac):**
```bash
export SAM2_CHECKPOINT="/path/to/project/checkpoints/sam2_hiera_base.pt"
```

**Example (Windows):**
```cmd
set SAM2_CHECKPOINT=C:\path\to\project\checkpoints\sam2_hiera_base.pt
```

## 4. Troubleshooting

If SAM2 is not found or the checkpoint is missing, the pipeline will automatically fall back to an Otsu thresholding method, ensuring the application continues to run (though with potentially less accurate segmentation for difficult images).
