# G2Net

This project involves training and predicting with a machine learning model using noisy signal data. The steps below outline the required environment setup, dependencies, and instructions for data preparation, model training, and result prediction.

## Environment Setup

Ensure the following tools are installed and configured in your environment:
- **CUDA**
- **Apex**

## Required Packages

Install the following Python packages before running the scripts:

```bash
pip install timm numpy omegaconf pandas pyfstat pytorch_lightning scikit_learn torch tqdm wandb
```

## Run Instructions

### Step 0: Download Raw Data
Download the raw data (approximately 200GB). This may take some time, so please be patient.

### Step 1: Generate Signal Images with Noise
Run the script to generate signal images by adding noise to clean signals.

```bash
python scripts/simulate_signals.py resources/competition/timestamps.pkl
```

### Step 2: Combine Gaussian Noise with Pure Signals
Generate random Gaussian background noise and combine it with pure signals.

```bash
python scripts/synthesize_external_psds.py resources/external/train/signals
```

### Step 3: Convert HDF5 Data Format
Convert HDF5 data files to the required input format for the model.

```bash
python extract_psds_from_hdf5.py ../input/train/test_hdf5_directory
```

### Step 4: Train the Model
Train the machine learning model using the specified configuration file.

```bash
python src/train.py config/convnext_small_in22ft1k.yaml
```

### Step 5: Predict Results
Use the trained model to generate predictions.

```bash
python src/predict.py convnext_small_in22ft1k-6f6648-last.pt --use-flip-tta
```

This will produce `submission1.csv`.

### Step 6: Train and Predict with Model 2
Train a second model and generate additional predictions.

```bash
python src/g2net-augmentation.py
```

This will produce `submission2.csv`.

### Step 7: Model Ensemble
Combine the results from both models for final submission.

```bash
python src/combine.py
```

This will generate the final `submission.csv`.

