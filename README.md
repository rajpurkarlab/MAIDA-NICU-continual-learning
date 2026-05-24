# Repurposing Adult AI Radiograph Interpretation Models for Neonatal Care Through Continual Learning: An International Multi-Center Study

This repository contains code for the study "Repurposing Adult AI Radiograph Interpretation Models for Neonatal Care Through Continual Learning: An International Multi-Center Study". Here, continual learning across multiple hospitals is used to adapt adult endotracheal tube (ETT) placement detection models to interpret neonatal chest X-rays from neonatal intensive care units (NICUs).

## Overview

This system implements continual learning methods to train ETT detection models sequentially across 30 hospitals from 20 countries and territories worldwide as well as compare against conventional single-site fine-tuning and inference directly from the initial adult model.

<img width="1265" height="528" alt="image" src="https://github.com/user-attachments/assets/99adc3b3-ba56-47ea-a39b-fca24f22f6dc" />

### Key Features

- **Multi-Hospital Continual Learning**: Sequential training across 30 international hospitals
- **Naive Fine-tuning**: Simple fine-tuning baseline
- **ETT & Carina Detection**: Localization of endotracheal tube tip and carina landmarks
- **Clinical Analysis Tools**: Precision/recall analysis and ETT placement classification
- **Leave-One-Out Validation**: Holdout analysis for generalization testing

## Quick Start

### 1. Install Environment

```bash
conda env create -f environment.yml
conda activate mrg
```

### 2. Download Pre-trained Model

Download `CL_pretrained.pth` [here](https://drive.google.com/file/d/1J2GlaqBNMuz0LgeamK50598hypiNf8sh/view?usp=sharing) and place it in `demo/models/`:

```
demo/models/CL_pretrained.pth
```

### 3. Run Demo

The demo performs **endotracheal tube (ETT) tip and carina localization** on chest X-rays. Place your images (PNG format) in `demo/input_images/`, then run:

```bash
python demo/run_demo.py
```

Results are saved to `demo/output/predictions.csv`, which contains the predicted (x, y) coordinates for the ETT tip and carina in the 640x640 preprocessed image space.

**Note**: To calculate the ETT tip-to-carina distance in centimeters for clinical triage, you will need to apply your own pixel spacing values from the original DICOM metadata.

---

## Using Your Own Data

### Data Format

#### Images
- Preprocessed to 640x640 pixels
- PNG format recommended
- Grayscale or RGB

#### Annotations (COCO Format)

Annotations follow the COCO object-detection schema. Image `id` is an integer
and is referenced by each annotation's `image_id`; the category ids are
`1 = tip`, `2 = carina` (the model uses only these two). A complete, correctly
formatted example is provided at [`data/example_annotations.json`](data/example_annotations.json).

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image_001.png",
      "width": 640,
      "height": 640,
      "hospital_name": "Hospital-A"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height]
    }
  ],
  "categories": [
    {"id": 1, "name": "tip"},
    {"id": 2, "name": "carina"}
  ]
}
```

### Preprocessing Your Data

Use the preprocessing script to resize and normalize your images:

```bash
# Edit paths in the script first
python preprocessing/preprocess_to_640x640.py
```

The preprocessing applies:
- Histogram equalization
- Resize to 640x640 (preserving aspect ratio with padding)
- Grayscale to RGB conversion

### Download Original CarinaNet Model Weights (Before Continual Learning)

Download the CarinaNet pretrained model from the official repository:
- Repository: [https://github.com/USM-CHU-FGuyon/CarinaNet](https://github.com/USM-CHU-FGuyon/CarinaNet)
- Place the downloaded model as: `models/CarinaNet/model.pt`

### Running Continual Learning

1. **Create a configuration file** based on `configs/continual_learning/config_naive.yaml`:

```yaml
data_path: '/path/to/your/preprocessed/images'
annos_dir: '/path/to/your/annotations'
train_annos_path: '/path/to/train-annotations.json'
test_annos_path: '/path/to/test-annotations.json'
output_path: '/path/to/output'

update_method: 'naive'           # continual-learning update method
number_of_simulation: 10         # Number of random hospital orderings
eval_current_hospital_only: true
wandb_off: true
```

2. **Run continual learning**:

```bash
python scripts/global_CL_sequential.py -c your_config.yaml
```

3. **Run holdout validation** (train on N-1 hospitals, test on held-out):

```bash
python scripts/global_CL_sequential_holdout.py \
    -c your_config.yaml \
    --holdout-hospital "Hospital-Name"
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `demo/run_demo.py` | Quick inference on new images |
| `scripts/global_CL_sequential.py` | Continual learning across hospitals |
| `scripts/global_CL_sequential_holdout.py` | Leave-one-out holdout validation |
| `scripts/global_single_hospital_ft.py` | Single-hospital fine-tuning baseline |
| `preprocessing/preprocess_to_640x640.py` | Image preprocessing |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code, please cite:

```
[Citation information to be added]
```

## Acknowledgments

- CarinaNet pretrained model: [https://github.com/USM-CHU-FGuyon/CarinaNet](https://github.com/USM-CHU-FGuyon/CarinaNet)
