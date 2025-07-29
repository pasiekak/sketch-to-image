# Sketch to Image - Pix2Pix GAN

Project implementing Pix2Pix GAN architecture for converting sketches to realistic images.

## 🚀 How to run the project

### 1. Activate virtual environment

```bash
# In the main project folder
.venv\Scripts\activate
```

### 2. Check environment

```bash
# Check Python version
python --version

# Check installed packages
pip list
```

### 3. Run training

```bash
# Run main script
python main.py
```

## 📈 Training monitoring

Generated images are saved every 10 epochs in folder:
`datasets/test/generated/`

## 🛠️ Troubleshooting

### Import issues

```bash
# Check if virtual environment is active
echo $VIRTUAL_ENV

# Install from requirements.txt
pip install -r requirements.txt

# Or install main packages manually
pip install tensorflow keras matplotlib opencv-python numpy
```

### GPU check (optional)

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 📝 Notes

-   Project uses `.venv` virtual environment - always activate before work
-   GAN parameters can be tuned in `scripts/config.py`
-   Training results are automatically saved
-   Create `requirements.txt` with: `pip freeze > requirements.txt`
