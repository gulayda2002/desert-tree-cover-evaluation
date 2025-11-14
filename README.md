# Desert Tree Cover Evaluation

A deep learning project for semantic segmentation of desert tree cover using satellite/aerial imagery. Compares **Random Forest** (pixel-based) and **U-Net** (deep learning) approaches for binary tree/non-tree classification.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This project implements and evaluates two machine learning approaches for detecting tree cover in desert environments:

- **Random Forest (RF):** Traditional machine learning with pixel-based RGB feature classification
- **U-Net:** Deep learning semantic segmentation with encoder-decoder architecture

### Key Results
- **Best U-Net Performance:** 47.02% test mIoU (51.16% validation mIoU)
- **Random Forest:** Consistent ~44% test mIoU across configurations
- **Dataset:** 2,100 images (256×256) split into train/val/test

## 📁 Repository Structure

```
desert-tree-cover-evaluation/
├── main.py              # Main training and evaluation script
├── run_all.sh          # Automated pipeline execution script
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── dataset/           # Training data (structure below)
│   ├── train/
│   │   ├── images/   # 700 training images
│   │   └── masks/    # Binary masks (0=background, 1=tree)
│   ├── val/
│   │   ├── images/   # 350 validation images
│   │   └── masks/    # Validation masks
│   └── test/
│       ├── images/   # 1,050 test images
│       └── masks/    # Test masks
└── output/            # Training outputs (timestamped directories)
    └── run-YYYYMMDD_HHMMSS/
        ├── metrics.json                    # Performance metrics
        ├── best_unet.pth                   # Best model checkpoint
        ├── comparison_val.png              # Validation metrics chart
        ├── comparison_test.png             # Test metrics chart
        ├── rf_val_confusion_matrix.png     # RF validation confusion matrix
        ├── rf_test_confusion_matrix.png    # RF test confusion matrix
        ├── unet_val_confusion_matrix.png   # U-Net validation confusion matrix
        ├── unet_test_confusion_matrix.png  # U-Net test confusion matrix
        └── sample_XX.png                   # Qualitative predictions (12 samples)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, 6GB+ VRAM)
- 16GB+ RAM

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/gulayda2002/desert-tree-cover-evaluation.git
cd desert-tree-cover-evaluation
```

2. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Prepare dataset:**
   - Place your images in `dataset/train/images/`, `dataset/val/images/`, `dataset/test/images/`
   - Place corresponding binary masks in `dataset/train/masks/`, `dataset/val/masks/`, `dataset/test/masks/`
   - Masks should be binary: 0 for background, 1 for tree pixels

### Running the Pipeline

**Automatic execution (recommended):**
```bash
./run_all.sh
```

This runs the complete pipeline with optimized settings:
- Random Forest with 20K pixel sampling per image
- U-Net with 60 epochs, batch size 8
- Full train/validation/test evaluation
- Generates all visualizations and metrics

**Custom execution:**
```bash
# Basic run
python main.py

# Custom Random Forest settings
python main.py --rf-max-pixels 30000 --rf-n-estimators 200 --rf-max-depth 30

# Custom U-Net settings
python main.py --epochs 80 --batch-size 16

# RF only (faster, no deep learning)
python main.py --rf-only

# Specify output directory
python main.py --outdir my_experiment_001

# Generate more qualitative samples
python main.py --viz-samples 20
```

## ⚙️ Configuration Options

### Command-Line Arguments

#### General Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--rf-only` | False | Skip U-Net training (RF only) |
| `--outdir` | `output/run-<timestamp>` | Output directory path |
| `--viz-samples` | 12 | Number of qualitative prediction samples |

#### Random Forest Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--rf-max-pixels` | 20000 | Pixels sampled per image for training |
| `--rf-n-estimators` | 150 | Number of trees in the forest |
| `--rf-max-depth` | 25 | Maximum tree depth |
| `--rf-min-samples` | 5 | Minimum samples required to split |

#### U-Net Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 60 | Number of training epochs |
| `--batch-size` | 8 | Training batch size |

### Environment Variables (run_all.sh)

You can customize the shell script by setting environment variables:

```bash
# Random Forest settings
export RF_MAX_PIXELS=25000
export RF_N_EST=200
export RF_MAX_DEPTH=30
export RF_MIN_SAMPLES=10

# U-Net settings
export EPOCHS=80
export BATCH_SIZE=16

# Then run
./run_all.sh
```

## 📊 Output Files

Each training run creates a timestamped directory in `output/` containing:

### Metrics File (`metrics.json`)
```json
{
  "rf": {
    "validation": {
      "OA": 0.9223,      // Overall Accuracy
      "mIoU": 0.5028,    // Mean Intersection over Union
      "Kappa": 0.6216    // Cohen's Kappa
    },
    "test": {
      "OA": 0.9074,
      "mIoU": 0.4409,
      "Kappa": 0.5478
    }
  },
  "unet": {
    "validation": {
      "OA": 0.9421,
      "mIoU": 0.5116,
      "Kappa": 0.6365
    },
    "test": {
      "OA": 0.9336,
      "mIoU": 0.4702,
      "Kappa": 0.5846
    }
  }
}
```

### Visualization Files (300 DPI, publication-ready)

1. **Comparison Charts:**
   - `comparison_val.png` - Bar chart comparing RF vs U-Net on validation set
   - `comparison_test.png` - Bar chart comparing RF vs U-Net on test set

2. **Confusion Matrices:**
   - `rf_val_confusion_matrix.png` - Random Forest validation performance
   - `rf_test_confusion_matrix.png` - Random Forest test performance
   - `unet_val_confusion_matrix.png` - U-Net validation performance
   - `unet_test_confusion_matrix.png` - U-Net test performance

3. **Qualitative Samples:**
   - `sample_00.png` to `sample_11.png` - Side-by-side comparison showing:
     - Original image
     - Ground truth mask
     - Random Forest prediction
     - U-Net prediction

### Model Checkpoint
- `best_unet.pth` - Best U-Net model weights (based on validation mIoU)

## 🏗️ Architecture Details

### U-Net Model
- **Input:** 256×256×3 RGB images
- **Output:** 256×256 binary segmentation masks
- **Architecture:** Standard encoder-decoder with skip connections
- **Training:** Mixed precision (FP16), gradient clipping, class-weighted loss
- **Augmentation:** Horizontal flip, vertical flip, 90° rotations

### Random Forest
- **Features:** RGB pixel values (3 features per pixel)
- **Sampling:** Balanced foreground/background sampling to handle class imbalance
- **Training:** Scikit-learn RandomForestClassifier with configurable hyperparameters

## �� Best Practices

### Memory Management

**For 12GB GPU:**
- U-Net batch size: 8-16
- Image size: 256×256

**For 6-8GB GPU:**
- U-Net batch size: 4-8
- Consider reducing to 128×128 if needed

**For Random Forest:**
- Use pixel sampling (20K-30K pixels per image)
- Balance foreground/background classes
- ~200MB memory for 700 training images with 20K sampling

### Performance Tips

1. **Data augmentation is critical:** Flips and rotations improve generalization significantly
2. **Class weights matter:** Trees typically occupy only ~10% of pixels
3. **Validation monitoring:** Save best model based on validation mIoU
4. **Simple configurations work best:** Standard Adam optimizer with lr=1e-3
5. **Patience with training:** 60 epochs with proper augmentation gives best results

## 📈 Expected Performance

Based on extensive experimentation (11 training runs):

| Metric | Random Forest | U-Net |
|--------|--------------|--------|
| **Test mIoU** | 43-45% | 7-47% |
| **Best Test** | 44.10% | 47.02% |
| **Best Validation** | 50.28% | 51.16% |
| **Consistency** | High (±1%) | Variable (depends on config) |

**Key Findings:**
- Random Forest provides stable baseline (~44% mIoU)
- U-Net achieves higher peak performance but is sensitive to hyperparameters
- Enhanced augmentation (flips + rotations) crucial for U-Net performance
- Standard hyperparameters outperform over-engineered solutions

### Hardware Used for Training

All experiments were conducted on the following hardware:
- **CPU:** AMD Ryzen 7 7700X (8 cores, 16 threads)
- **GPU:** NVIDIA RTX 3060 (12GB VRAM)
- **RAM:** 32GB DDR5
- **Training Time:** ~6 hours total for 11 runs (average ~30-35 minutes per run with 60 epochs)

## 🔧 Troubleshooting

### Out of Memory (OOM) Errors

**Random Forest OOM:**
```bash
# Reduce pixel sampling
python main.py --rf-max-pixels 10000

# Reduce forest size
python main.py --rf-n-estimators 100 --rf-max-depth 20
```

**U-Net OOM:**
```bash
# Reduce batch size
python main.py --batch-size 4

# Or train RF only
python main.py --rf-only
```

### Poor Performance

1. **Check data quality:** Ensure masks are binary (0 and 1 only)
2. **Verify dataset structure:** Images in `images/`, masks in `masks/`
3. **Enable augmentation:** Default configuration includes augmentation
4. **Monitor validation:** Check if validation mIoU improves during training

### CUDA Errors

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If CUDA unavailable, code will automatically use CPU (slower)
```

## 📚 Citation

If you use this code in your research, please cite:

**Software:**
```bibtex
@software{desert_tree_cover_2025,
  author = {Kuandikova, Gulayda and Mamataliev, Abror},
  title = {Desert Tree Cover Evaluation},
  year = {2025},
  url = {https://github.com/gulayda2002/desert-tree-cover-evaluation}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

**Authors:**
- Gulayda Kuandikova - gulaydakuandikova028@gmail.com
- Abror Mamataliev - abrornomos2018@gmail.com

For questions or issues, please:
- Open an issue on GitHub
- Contact the authors via email

## 🙏 Acknowledgments

This work builds upon and utilizes the following resources:

### Dataset
- **High-resolution UAV Desert Vegetation Dataset**  
  Figshare: https://figshare.com/articles/dataset/High-resolution_UAV_desert_vegetation_dataset/28387949

### Related Research
- Hua, S., Yang, B., Zhang, X., Qi, J., Su, F., Sun, J., & Ruan, Y. (2025). **GDPGO-SAM: An Unsupervised Fine Segmentation of Desert Vegetation Driven by Grounding DINO Prompt Generation and Optimization Segment Anything Model.** *Remote Sensing*, 17(4), 691.  
  DOI: https://doi.org/10.3390/rs17040691

### Software and Tools
- U-Net architecture based on the original paper by Ronneberger et al. (2015)
- PyTorch team for the excellent deep learning framework
- Scikit-learn for Random Forest implementation

---

**Project Status:** ✅ Complete  
**Last Updated:** November 14, 2025  
**Version:** 1.0.0
