# This folder contains dataset structures for machine learning training

## Structure:

-   `boots/` - Boot images dataset
-   `faces/` - Face images dataset
-   `faces_all/` - Combined face datasets
-   `faces_augmented/` - Augmented face datasets
-   `faces_men/` - Male faces dataset
-   `faces_women/` - Female faces dataset
-   `temp/` - Temporary processing files

## Typical folder structure:

```
dataset/
├── dataset_name/
│   ├── train/
│   │   ├── original/    # Original images
│   │   ├── sketch/      # Corresponding sketches
│   │   └── augmented/   # (Optional) Augmented sketches
│   ├── test/
│   │   ├── original/
│   │   └── sketch/
│   └── validation/
│       ├── original/
│       └── sketch/
```

Place your training datasets in the appropriate subfolders.
