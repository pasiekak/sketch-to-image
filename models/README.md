# Models Directory

This directory contains trained models and experiment results.

## Structure:

Each experiment creates a folder with naming convention:
`#DATASET_S{samples}_E{epochs}_B{batch}_LR{learning_rate}_...`

Example experiment folder structure:

```
models/
├── #FACES_ALL_S20000_E100_B1_LR0.0001_.../
│   ├── d_model.keras           # Final discriminator model
│   ├── g_model.keras           # Final generator model
│   ├── sketch.png              # Sample input sketch
│   ├── original.png            # Sample target image
│   ├── summary_last_epoch.png  # Final training summary
│   ├── training_statistics.csv # Training metrics
│   ├── models/                 # Model checkpoints
│   │   ├── g_model_epoch_001.keras
│   │   ├── d_model_epoch_002.keras
│   │   └── ...
│   ├── generated/              # Sample outputs per epoch
│   │   ├── generated_epoch_001.png
│   │   ├── generated_epoch_002.png
│   │   └── ...
│   └── predictions/            # Generated predictions
│       ├── image1_g_model_O.jpg
│       ├── image1_g_model_S.jpg
│       ├── image1_g_model_P.jpg
│       └── ...
```

Models are automatically saved here during training.
