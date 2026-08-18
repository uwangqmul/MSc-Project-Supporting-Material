# MusicGen-small Erhu LoRA Adaptation

## 1. Project Overview
This repository provides an end-to-end, reproducible framework for fine-tuning Meta's `facebook/musicgen-small` model using Low-Rank Adaptation (LoRA). The objective is to adapt the base model to the traditional Chinese two-stringed bowed instrument, the Erhu. 

The entire pipeline—from data preprocessing to model evaluation—is encapsulated in a single Jupyter Notebook optimized for execution on a Google Colab T4 GPU environment.

## 2. Core Components
* **`MSc814_YouWang.ipynb`**: The primary execution notebook. It contains all logic for environment setup, data preprocessing, model training, and evaluation. It is fully compatible with Google Colab and can be executed end-to-end sequentially.
* **`CCOM-HuQin-v2.0.1-audios.zip`**: The underlying Chinese instrument music dataset used for training.
* **`originalmusic/`**: The specific directory extracted from the dataset containing the isolated Erhu audio files.
* **`erhu_lora_runs/`**: The automated output tracking directory where all experimental logs, model checkpoints, and evaluation artifacts are stored.

## 3. Environment & Prerequisites
To run this project, a Google Colab instance with a **T4 GPU** is recommended. The notebook automatically installs the required dependencies:

```bash
pip install transformers==4.46.3 peft==0.13.2 accelerate>=0.34,<2 soundfile sentencepiece frechet-audio-distance==0.3.4 pandas scipy
```

## 4. Dataset Workflow
The data pipeline is designed to ensure strict machine learning hygiene:
1. **Extraction & Filtering:** Isolates the Erhu subset (`originalmusic`) from the `CCOM-HuQin-v2.0.1-audios.zip` archive.
2. **Audio Normalization:** All audio is standardized and resampled to 32 kHz mono to match MusicGen's expected input.
3. **Canonical-Piece Splitting:** Implements a strict deterministic split. Multiple recordings or variations of the same musical piece are kept within the same split to strictly prevent data leakage between training, validation, and test sets.
4. **Segmentation:** Fixed-length clips are generated using random cropping during the training phase and deterministic center cropping during evaluation.

## 5. Execution Guide
1. Open `MSc814_YouWang.ipynb` in Google Colab.
2. Ensure the hardware accelerator is set to **T4 GPU** (`Runtime` > `Change runtime type`).
3. Upload or mount `CCOM-HuQin-v2.0.1-audios.zip` to the Colab environment.
4. Run all cells sequentially (`Runtime` > `Run all`). 
5. The notebook will automatically handle extraction, dataset creation, training loops, and metric evaluations.

## 6. Outputs & Artifacts (`erhu_lora_runs/`)
As the notebook runs, it generates rich telemetry and artifacts stored within the `erhu_lora_runs/` directory. This directory includes:
* **Training Logs:** Epoch-by-epoch loss curves, hyperparameter configurations, and validation metrics.
* **Model Checkpoints:** Saved LoRA adapter weights for different rank sweeps, optimized via early stopping mechanisms.
* **Data Manifests:** Traceable JSON/CSV files documenting the exact data splits.
* **Evaluation Outputs:** Generated sample `.wav` files and quantitative metric reports.

## 7. Evaluation & Metrics
* **Configurable Sweeps:** The framework supports automated sweeping over different LoRA ranks to find the optimal parameter efficiency.
* **Standardization:** Generation relies on standardized prompts, fixed random seeds, and consistent temperature/top-k parameters to ensure fair comparisons.
* **Objective Scoring:** Evaluates the acoustic quality and domain similarity using **Fréchet Audio Distance (FAD)** scoring.

---
*Disclaimer: The specific playing technique annotations provided with the CCOM-HuQin dataset are not utilized in this project. The adaptation focuses on overall Erhu timbral and domain similarity.*
