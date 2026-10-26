# AdaTM: Adaptive Fusion of Transformer and Mamba for Time Series Forecasting

This repository provides the implementation of **AdaTM (Adaptive Fusion of Transformer and Mamba for Time Series Forecasting)**.

AdaTM combines multi-level discrete wavelet transform (DWT), resolution-specific patching, and adaptive fusion of **Transformer** and **bidirectional Mamba (Bi-Mamba)** experts for time-series forecasting.

**Code repository:** https://github.com/wwy-1208/AdaTM

> **Manuscript-aligned configuration.** The experimental settings below follow the revised AdaTM manuscript and its response letter for *Digital Signal Processing* (DSP-D-26-02831). In the manuscript, each dataset has **one fixed wavelet/patch/training configuration across all four prediction horizons**. Accordingly, this README does not use horizon-specific hyperparameter overrides. These settings document the manuscript's stated experimental protocol; the exact correspondence of published results to successful execution logs and checkpoints should be verified independently before claiming exact reproduction.

## 1. Overview

AdaTM contains three primary components:

1. **Multi-level wavelet decomposition:** DWT decomposes the input sequence into one approximation coefficient sequence and several detail coefficient sequences, allowing the model to analyze multiple temporal resolutions.
2. **Resolution-specific patching:** Each wavelet branch uses its own *predefined* patch length and stride to construct patch tokens.
3. **Adaptive feature fusion:** Transformer and Bi-Mamba experts encode complementary temporal information. A learned softmax gate dynamically combines their representations before forecasting.

The term **adaptive** refers to the learned expert-fusion weights. The wavelet basis, DWT decomposition level, patch lengths, and strides are selected as experimental hyperparameters; they are **not learned or changed automatically for individual inputs**.

## 2. Repository structure

```text
AdaTM/
├── run.py                    # Training and testing entry point
├── models/                   # AdaTM model
├── exp/                      # Experiment and optimization pipeline
├── layers/                   # Transformer, Mamba, DWT, patch, fusion layers
├── data_provider/            # Dataset loaders and preprocessing
├── utils/                    # Evaluation metrics and training utilities
├── pytorch_wavelets/         # Bundled wavelet implementation (where present)
├── scripts/                  # Experiment scripts (verify against this README)
└── data/                     # Local benchmark datasets
```

The existing command-line model identifier is `--model AdaTM`. The directory names shown above refer to the repository layout documented with this codebase; inspect the checkout if you are using a different code revision.

## 3. Environment

A Linux machine with an NVIDIA GPU is recommended. The package builds of **PyTorch**, **CUDA**, and **mamba-ssm** must be compatible.

```bash
conda create -n adatm python=3.10 -y
conda activate adatm

# Install a CUDA-enabled PyTorch build compatible with your system:
# https://pytorch.org/get-started/locally/

pip install numpy pandas scipy scikit-learn matplotlib einops PyWavelets \
  transformers ptflops yacs

# Install mamba-ssm after selecting compatible PyTorch / CUDA versions:
pip install mamba-ssm
```

The manuscript does not specify an exact fully pinned package environment. For a reproducible release, record the specific Python, PyTorch, CUDA, Mamba, and wavelet-library versions tested with the code. If the repository bundles `pytorch_wavelets/`, retain the implementation used in the successful runs.

## 4. Datasets

Arrange the seven forecasting datasets as follows:

```text
data/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── weather/
│   └── weather.csv
├── electricity/
│   └── electricity.csv
└── traffic/
    └── traffic.csv
```

| Dataset | Variables | Sampling frequency |
|---|---:|---|
| ETTh1 | 7 | Hourly |
| ETTh2 | 7 | Hourly |
| ETTm1 | 7 | 15 minutes |
| ETTm2 | 7 | 15 minutes |
| Weather | 21 | 10 minutes |
| Electricity | 321 | Hourly |
| Traffic | 862 | Hourly |

The manuscript uses **multivariate-to-multivariate forecasting**, corresponding to `--features M` in the code. Its forecasting horizons are

\[
F \in \{96, 192, 336, 720\}.
\]

The look-back length is **96 for ETTh1** and **672 for the other six datasets**. All model comparisons must use the data partition and scaling protocol reported in the manuscript. The loader's actual split and column requirements depend on the code revision. For the provided code conventions, ETT datasets use their standard chronological partitions; the generic `custom` loader uses chronological partitions and generally requires a `date` field and the designated target column (commonly `OT`). Check your actual CSV column names before executing the examples below.

## 5. Experimental configurations reported in the paper

### 5.1 Dataset-specific configurations

**Table 1.** Dataset-specific settings used for the four prediction horizons **96, 192, 336, and 720**. The DWT and patch configurations are held fixed **within each dataset** across these horizons.

| Dataset | Look-back `seq_len` | Batch size | Max learning rate | Wavelet | DWT level | Patch lengths | Strides |
|---|---:|---:|---:|---|---:|---|---|
| ETTh1 | 96 | 256 | `3e-4` | `db3` | 2 | `2,4,4` | `1,2,2` |
| ETTh2 | 672 | 256 | `3e-4` | `coif5` | 3 | `8,16,32,48` | `4,8,16,24` |
| ETTm1 | 672 | 128 | `3e-4` | `db2` | 1 | `8,48` | `4,24` |
| ETTm2 | 672 | 128 | `3e-4` | `db2` | 1 | `8,48` | `4,24` |
| Weather | 672 | 128 | `3e-4` | `db3` | 2 | `4,8,16` | `2,4,8` |
| Electricity | 672 | 16 | `3e-4` | `sym3` | 1 | `8,16` | `4,8` |
| Traffic | 672 | 16 | `1e-4` | `db3` | 1 | `4,24` | `2,12` |

**Important:** The manuscript reports one dataset-specific configuration for every prediction horizon. In particular, **do not switch ETTh2 from `coif5` to `db3`, do not change Weather from DWT level 2 to level 1, and do not change Electricity from `sym3` to another basis for longer horizons** if your goal is to follow the revised manuscript protocol.

### 5.2 Shared architecture and training settings

**Table 2.** Fixed settings as stated in the revised manuscript.

| Parameter | Manuscript setting |
|---|---|
| Optimizer | Adam |
| Learning-rate scheduler | OneCycleLR |
| Maximum training epochs | **10** |
| Training loss | MSE |
| Early stopping | Validation loss, patience = 5 |
| Main experiment random seed | 2021 |
| Five-seed statistical experiments | 2021, 2022, 2023, 2024, 2025 |
| Embedding dimension | **32 for all datasets** |
| Transformer encoder layers | 3 |
| Attention heads | 8 |
| Feed-forward dimension | 128 |
| Mamba layers (manuscript configuration) | 2 |
| Mamba expansion factor | 1 |
| Mamba state dimension | 16 |
| Mamba convolution kernel size | 4 |
| Activation | GELU |
| RevIN | Enabled |
| RevIN affine transformation | Enabled |
| Dropout | **0.2 for all datasets** |
| Fully connected dropout | **0.2 for all datasets** |
| Initialization | PyTorch defaults |
| Weight decay | 0 |

The revised LaTeX configuration table also records a decoder-layer configuration value of `1`. The provided README for the codebase notes that forecasting is performed through a flattening prediction head rather than an active Transformer decoder. Likewise, the command-line fields `--e_layers` and `--m_layers` may describe the construction of MoE blocks and Bi-Mamba components rather than a simple count of stacked vanilla Transformer/Mamba layers. **For structural reproducibility, compare the model graph in the exact code revision with the manuscript's architecture table**; documenting a flag alone does not prove the stated number of active layers.

The manuscript does **not** explicitly state `label_len`, OneCycleLR warm-up fraction, or all framework-level default arguments. The runnable examples below use `label_len = seq_len / 2` (`48` or `336`) following the previously documented code convention. This is an **example execution choice, not a manuscript-verified hyperparameter**; check the original successful experiment logs when exact numerical reproduction is required.

## 6. Running AdaTM

Run the following commands from the repository root. In the codebase described by the original README, `--is_training 1` trains the model and then evaluates the final experiment, and `--model AdaTM` selects the AdaTM implementation.

### 6.1 Example: ETTh1, prediction length 96

```bash
python -u run.py \
  --is_training 1 --model AdaTM --model_id AdaTM_ETTh1_96_96 \
  --data ETTh1 --root_path ./data/ETT-small/ --data_path ETTh1.csv \
  --features M --target OT --freq h \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --wavelet db3 --level 2 \
  --patch_len_ls 2,4,4 --stride_ls 1,2,2 \
  --batch_size 256 --learning_rate 0.0003 --train_epochs 10 \
  --d_model 32 --n_heads 8 --e_layers 3 --d_layers 1 \
  --m_layers 2 --d_ff 128 --d_state 16 --d_conv 4 \
  --dropout 0.2 --fc_dropout 0.2 --patience 10 \
  --random_seed 2021 --gpu 0
```

### 6.2 Example: ETTh2, prediction length 720

The ETTh2 parameters **do not change** when the prediction length increases from 96 to 720:

```bash
python -u run.py \
  --is_training 1 --model AdaTM --model_id AdaTM_ETTh2_672_720 \
  --data ETTh2 --root_path ./data/ETT-small/ --data_path ETTh2.csv \
  --features M --target OT --freq h \
  --seq_len 672 --label_len 336 --pred_len 720 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --wavelet coif5 --level 3 \
  --patch_len_ls 8,16,32,48 --stride_ls 4,8,16,24 \
  --batch_size 256 --learning_rate 0.0003 --train_epochs 10 \
  --d_model 32 --n_heads 8 --e_layers 3 --d_layers 1 \
  --m_layers 2 --d_ff 128 --d_state 16 --d_conv 4 \
  --dropout 0.2 --fc_dropout 0.2 --patience 10 \
  --random_seed 2021 --gpu 0
```


## 7. Acknowledgements

The implementation builds upon ideas or tooling associated with [SST](https://arxiv.org/abs/2404.14757), [Mamba](https://github.com/state-spaces/mamba), [PatchTST](https://github.com/yuqinie98/PatchTST), and [Autoformer](https://github.com/thuml/Autoformer). We thank the original authors and maintainers.


