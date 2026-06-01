# Data Directory

The patient images, clinical metadata, and full annotation files used in the
study are **not distributed with this repository**. Chest radiographs and the
associated clinical data are governed by institutional data-sharing agreements
and cannot be released publicly.

To make the expected input format unambiguous, this directory includes a single
**format-only example**:

- `example_annotations.json` — a synthetic, two-image COCO annotation file that
  exactly matches the schema this pipeline expects. All coordinates and
  identifiers are fabricated; it contains no patient data. Use it as a
  structural reference when formatting your own annotations.

### Expected Directory Structure

To run the experiments on your own data, populate `data/` as follows (the file
counts scale with the number of hospitals you include):

```
data/
├── README.md                                  # This file
├── example_annotations.json                   # Provided — format reference only
│
├── images/                                     # Provide your own
│   ├── original/
│   │   └── hospitals/                          # Original-resolution images
│   └── preprocessed_640x640/
│       └── hospitals/                          # 640x640 images from the preprocessing script
│
├── annotations/                                # Provide your own
│   ├── original/                               # Original-resolution COCO annotations
│   │   ├── <Hospital>-train-annotations.json
│   │   ├── <Hospital>-test-annotations.json
│   │   └── ...
│   │
│   └── preprocessed_640x640/                   # 640x640 COCO annotations
│       ├── <Hospital>-train-annotations.json
│       ├── <Hospital>-test-annotations.json
│       ├── ...
│       ├── hospital-train-annotations.json     # Combined train set (all hospitals)
│       └── hospital-test-annotations.json      # Combined test set (all hospitals)
```

### Annotation Format

Annotations follow the COCO object-detection schema. Each file contains
`info`, `licenses`, `images`, `annotations`, and `categories` blocks. The
categories are:

| id | name | used by model |
|----|------|---------------|
| 1 | tip | yes (ETT tip) |
| 2 | carina | yes |
| 3 | top_thoracic_vertebra | no (reference only) |
| 4 | bottom_thoracic_vertebra | no (reference only) |

Only the ETT tip (category 1) and carina (category 2) are used during training
and evaluation; the two thoracic-vertebra landmarks are recorded for reference
but are not consumed by the model. See `example_annotations.json` for a complete,
correctly-formatted example.
