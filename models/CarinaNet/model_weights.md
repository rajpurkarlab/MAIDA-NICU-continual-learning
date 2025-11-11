# CarinaNet Pretrained Model Weights

## Download Instructions

The pretrained CarinaNet model weights are required to run continual learning experiments. Due to file size constraints (139 MB), the weights are not included in this repository.

### Download from CarinaNet Repository

1. **Download the pretrained model** from the official CarinaNet repository:
   - Repository: [https://github.com/USM-CHU-FGuyon/CarinaNet](https://github.com/USM-CHU-FGuyon/CarinaNet)
   - Direct link: [https://github.com/USM-CHU-FGuyon/CarinaNet/tree/master/TRAI_ICU](https://github.com/USM-CHU-FGuyon/CarinaNet/tree/master/TRAI_ICU)

2. **Rename the downloaded file** to `model.pt`

3. **Place the file** in this directory:
   ```
   models/CarinaNet/model.pt
   ```

## Expected File Structure

After downloading, your directory should look like:
```
models/
├── __init__.py
├── ETTModel.py
└── CarinaNet/
    ├── __init__.py
    ├── CarinaNetModel.py
    ├── model.pt          <- Downloaded pretrained weights
    └── retinanet/
        ├── ...
```

## Model Details

- **Architecture**: RetinaNet with ResNet-50 backbone
- **Task**: Endotracheal tube (ETT) tip and carina detection
- **Input**: 640x640 RGB chest X-ray images
- **Output**: Bounding boxes for ETT tip (class 0) and carina (class 1)
- **File Size**: ~139 MB

## Verification

To verify the model was downloaded correctly, run:
```python
import torch

model_path = "models/CarinaNet/model.pt"
try:
    model = torch.load(model_path, map_location='cpu')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
```

## Citation

If you use the pretrained CarinaNet model, please cite:
```
Oliver M, Renou A, Allou N, Moscatelli L, Ferdynus C, Allyn J. Image augmentation and automated measurement of endotracheal-tube-to-carina distance on chest radiographs in intensive care unit using a deep learning model with external validation. Crit Care. 2023 Jan 25;27(1):40. doi: 10.1186/s13054-023-04320-0. PMID: 36698191; PMCID: PMC9878756.
```
