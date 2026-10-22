# AdaTM: Adaptive Fusion of Transformer and Mamba for Time Series Forecasting

Official implementation of **AdaTM (Adaptive Fusion of Transformer and Mamba for Time Series Forecasting)**.

AdaTM combines **multi-level discrete wavelet decomposition (DWT)**, **resolution-specific patch embedding**, and **adaptive fusion of Transformer and bidirectional Mamba experts** for multivariate time-series forecasting. This repository contains the model, data loaders, training/evaluation code, and experimental settings used in our study.

**Repository:** https://github.com/wwy-1208/AdaTM

## Overview

1. **Multi-resolution decomposition:** DWT separates each input series into approximation and detail coefficients.
2. **Resolution-specific patching:** Each coefficient branch uses specified patch lengths and strides.
3. **Hybrid experts:** Transformer and bidirectional Mamba modules learn complementary temporal representations; a learned gate fuses their outputs.
4. **Forecasting:** Branch representations are combined and projected to the requested prediction horizon.

**Scope of adaptivity.** AdaTM learns the expert-fusion weights. The wavelet basis, DWT level, and branch patch lengths/strides are **configured before training**, not automatically selected for each sample.

## Repository layout

```text
AdaTM/
├── run.py                       # Training and evaluation entry point
├── exp/exp_main.py              # Training, validation and testing
├── models/AdaTM.py                # AdaTM model implementation (class AdaTMModel)
├── layers/                      # Wavelet, patching, Transformer and Mamba modules
├── data_provider/               # Dataset loaders and preprocessing
├── utils/                       # Metrics, early stopping and helper functions
├── pytorch_wavelets/            # Wavelet implementation included in the source tree
├── scripts/                     # Legacy AdaTM scripts; see note below
└── data/                        # Place the datasets here
```

> **Entry-point naming:** In the provided code, AdaTM is implemented as `AdaTMModel` in `models/AdaTM.py`, and `run.py` registers it under `--model AdaTM`. **Use `--model AdaTM` in commands** unless you also update the model registry. Some files retain names inherited from the AdaTM codebase.
>
> **Script compatibility:** Some older files in `scripts/` use legacy argument names. The commands below use arguments supported by the current `run.py` and should be preferred when reproducing AdaTM experiments.

## Environment setup

A Linux machine with an NVIDIA CUDA GPU is recommended. The original project environment used Python 3.10 and the `mamba-ssm` package. PyTorch, CUDA, and `mamba-ssm` versions must be mutually compatible; install a CUDA-enabled PyTorch build appropriate for your system before installing Mamba.

```bash
conda create -n adatm python=3.10 -y
conda activate adatm

# Install the CUDA-enabled PyTorch build compatible with your GPU/CUDA setup.
# Follow https://pytorch.org/get-started/locally/

pip install numpy pandas scipy scikit-learn matplotlib einops PyWavelets \
  transformers ptflops yacs

# Install a compatible mamba-ssm version for your PyTorch/CUDA environment.
# See https://github.com/state-spaces/mamba
pip install mamba-ssm
```

The source archive already contains a local `pytorch_wavelets/` package. If running a different code snapshot without that directory, install a compatible `pytorch-wavelets` implementation. Package versions have not been locked in this README; **record your actual Python, PyTorch, CUDA, and Mamba versions when reporting reproduced results**.

## Data preparation

Use the following layout (relative to the repository root):

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

The ETT loaders use the standard fixed chronological partitions. For datasets passed with `--data custom`, the loader uses chronological splits of approximately **70% training / 10% validation / 20% testing**. Standardization is fitted **only on training data** and then applied to the other splits. CSV files should have a `date` column; the current custom loader also expects a target column named `OT` (even when `--features M` is used).

In the reported multivariate setting, `--features M` forecasts all feature channels. Dataset channel counts in the provided CSV files are ETTh1/ETTh2/ETTm1/ETTm2: **7**, Weather: **21**, Electricity: **321**, and Traffic: **862**.

## Experimental configurations

### Dataset-specific settings (prediction horizon = 96)

The following table reproduces the representative dataset-specific settings reported in the manuscript (the seven-row configuration table). `L` denotes the look-back length; `label_len` is the history length passed to the data loader.

| Dataset | L | `label_len` | Channels | Batch size | Learning rate | Wavelet | DWT level | Patch lengths | Strides |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| ETTh1 | 96 | 48 | 7 | 256 | 3e-4 | db3 | 2 | 2,4,4 | 1,2,2 |
| ETTh2 | 672 | 336 | 7 | 256 | 3e-4 | coif5 | 3 | 8,16,32,48 | 4,8,16,24 |
| ETTm1 | 672 | 336 | 7 | 128 | 3e-4 | db2 | 1 | 8,48 | 4,24 |
| ETTm2 | 672 | 336 | 7 | 128 | 3e-4 | db2 | 1 | 8,48 | 4,24 |
| Weather | 672 | 336 | 21 | 128 | 3e-4 | db3 | 2 | 4,8,16 | 2,4,8 |
| Electricity | 672 | 336 | 321 | 16 | 3e-4 | sym3 | 1 | 8,16 | 4,8 |
| Traffic | 672 | 336 | 862 | 16 | 1e-4 | db3 | 1 | 4,24 | 2,12 |

**Important:** The table is **not** a single configuration applied unchanged to all four forecasting horizons. Several reported runs used horizon-specific settings. The adjustments documented in the original experiment commands are listed below.

<details>
<summary><strong>Horizon-dependent overrides from the recorded experiment commands</strong></summary>

| Dataset / horizon | Recorded changes relative to the horizon-96 row |
|---|---|
| ETTh1 / 192, 336, 720 | Same DWT/patch configuration and `L=96`, `label_len=48` |
| ETTh2 / 192 and 336 | db3, DWT level 1, patches `8,48`, strides `4,24`, 10 epochs |
| ETTh2 / 720 | db3, level 1, patches `48,8`, strides `24,4`, LR `1e-4`, dropout `0.3`, 10 epochs |
| ETTm1 / 192, 336, 720 | Same configuration as horizon 96, 10 epochs |
| ETTm2 / 192 and 336 | Same configuration as horizon 96, 10 epochs |
| ETTm2 / 720 | Batch size 64, LR `1e-4`, 10 epochs |
| Weather / 192 | db3, level 1, patches `8,24`, strides `4,12` |
| Weather / 336 | db3, level 1, patches `4,24`, strides `2,12` |
| Weather / 720 | Same as Weather/336, LR `1e-4` |
| Electricity / 192 | coif5, level 2, patches `4,8,16`, strides `2,4,8` |
| Electricity / 336 and 720 | sym4, level 2, patches `4,8,16`, strides `2,4,8` |
| Traffic / 192, 336, 720 | Same configuration as horizon 96 |

**Recorded-command caveats:** In the provided run-command document, the Electricity/192 command omits `--seq_len` and `--label_len` (the parser defaults would therefore be 96 and 48); the Electricity/96 command has a typo in `--train_epochs10` (correct syntax: `--train_epochs 10`); and the first ETTh1/96 command uses `--model AdaTM`, although the provided registry accepts `AdaTM`. Check successful experiment logs before asserting those ambiguous commands exactly reproduce published numbers.

</details>

### Architecture and training settings

| Parameter | Setting |
|---|---|
| Optimizer | Adam (`betas=(0.9, 0.999)`, `eps=1e-8`, weight decay `0`) |
| Learning-rate scheduler | OneCycleLR, stepped once per training mini-batch; default cosine annealing |
| OneCycle warm-up (`pct_start`) | 0.3, or 0.2 for Traffic |
| Maximum training epochs | 10 for most recorded runs; 20 in the recorded ETTh1 and ETTh2/96 runs |
| Loss | MSE with numerical clipping to `[-1e4, 1e4]` in `SafeMSELoss` |
| Early stopping | Validation loss; patience 10 epochs |
| Main random seed | 2021; five-seed robustness experiments: 2021, 2022, 2023, 2024, 2025 |
| Embedding dimension (`d_model`) | 32; Traffic uses 64 |
| MoE blocks per wavelet branch (`e_layers`) | 3 |
| Attention heads (`n_heads`) | 8 |
| Transformer layers in each expert | 1 encoder layer |
| Bidirectional Mamba expert | Forward and backward Mamba modules; one internal encoder layer per expert |
| Mamba expansion factor | 1 (set by the expert implementation) |
| Mamba state dimension (`d_state`) | 16 |
| Mamba convolution width (`d_conv`) | 4 |
| Feed-forward dimension (`d_ff`) | 128 |
| Activation | GELU |
| Main dropout | 0.2; selected runs use 0.3 (including Traffic) |
| FC dropout CLI setting | 0.2 by default (0.3 in specified runs); note that an exposed flag is not necessarily used by every active layer |
| RevIN / affine | Enabled / enabled |
| Parameter initialization | Module-specific PyTorch defaults and Mamba implementation defaults; RevIN scale initialized to 1, bias to 0 |
| Prediction head | Flattening-based head (not a separate Transformer decoder) |

**Terminology note:** The parser exposes `--m_layers 2` and `--d_layers 1`, but these flags should **not** be interpreted as two stacked Mamba encoder layers and an active Transformer decoder. The active implementation constructs three MoE blocks per branch, with one internal bidirectional Mamba expert layer in each MoE block.

## Running AdaTM

Execute commands from the repository root. `--is_training 1` **trains and then automatically evaluates** a run. The currently registered model identifier is `AdaTM`, even though the architecture used is AdaTM.

### Example 1: ETTh1 (`L=96`, `F=96`)

```bash
python -u run.py \
  --is_training 1 --model AdaTM --model_id AdaTM_ETTh1_96_96 \
  --data ETTh1 --root_path ./data/ETT-small/ --data_path ETTh1.csv \
  --features M --target OT --freq h \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --wavelet db3 --level 2 \
  --patch_len_ls 2,4,4 --stride_ls 1,2,2 \
  --batch_size 256 --learning_rate 0.0003 --train_epochs 20 \
  --d_model 32 --n_heads 8 --e_layers 3 --d_ff 128 \
  --d_state 16 --d_conv 4 \
  --dropout 0.2 --patience 10 \
  --random_seed 2021 --gpu 0
```

### Example 2: ETTh2 (`L=672`, `F=96`)

```bash
python -u run.py \
  --is_training 1 --model AdaTM --model_id AdaTM_ETTh2_672_96 \
  --data ETTh2 --root_path ./data/ETT-small/ --data_path ETTh2.csv \
  --features M --target OT --freq h \
  --seq_len 672 --label_len 336 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --wavelet coif5 --level 3 \
  --patch_len_ls 8,16,32,48 --stride_ls 4,8,16,24 \
  --batch_size 256 --learning_rate 0.0003 --train_epochs 20 \
  --d_model 32 --n_heads 8 --e_layers 3 --d_ff 128 \
  --d_state 16 --d_conv 4 \
  --dropout 0.2 --patience 10 \
  --random_seed 2021 --gpu 0
```

### Example 3: Weather (`L=672`, `F=96`)

```bash
python -u run.py \
  --is_training 1 --model AdaTM --model_id AdaTM_Weather_672_96 \
  --data custom --root_path ./data/weather/ --data_path weather.csv \
  --features M --target OT --freq h \
  --seq_len 672 --label_len 336 --pred_len 96 \
  --enc_in 21 --dec_in 21 --c_out 21 \
  --wavelet db3 --level 2 \
  --patch_len_ls 4,8,16 --stride_ls 2,4,8 \
  --batch_size 128 --learning_rate 0.0003 --train_epochs 10 \
  --d_model 32 --n_heads 8 --e_layers 3 --d_ff 128 \
  --d_state 16 --d_conv 4 \
  --dropout 0.2 --patience 10 \
  --random_seed 2021 --gpu 0
```

For **ETTm1**, **ETTm2**, **Electricity**, and **Traffic**, use the corresponding dataset row in the table above with `--data ETTm1` / `ETTm2` (ETT series) or `--data custom` (Electricity/Traffic), and set `--root_path`, `--data_path`, `--enc_in`, `--dec_in`, and `--c_out` accordingly. Use the horizon-specific overrides when changing `--pred_len`.

For **Traffic**, additionally set `--d_model 64 --dropout 0.3 --fc_dropout 0.3 --pct_start 0.2`. For a multi-GPU setup supported by the environment, the current code exposes `--use_multi_gpu --devices 0,1,2,3`.

### Multiple prediction horizons and seeds

To change horizons, set `--pred_len` to one of `96`, `192`, `336`, or `720` and apply the corresponding dataset/horizon-specific overrides above. To run the five-seed analysis, execute the same configuration separately with:

```bash
--random_seed 2021
--random_seed 2022
--random_seed 2023
--random_seed 2024
--random_seed 2025
```

Report results from actual completed runs; the README does not substitute for saved command logs and checkpoints.

## Output files

With `--is_training 1`, the entry point runs training followed by testing. The code writes artifacts under locations such as:

```text
checkpoints/<experiment_setting>/checkpoint.pth
results/<experiment_setting>/metrics.npy
results/<experiment_setting>/pred.npy
results/<experiment_setting>/true.npy
test_results/<experiment_setting>/
```

The console also reports MSE and MAE. The checkpoint/experiment setting is generated internally by `run.py`; it includes model/dataset and sequence configuration fields. **Testing with `--is_training 0` in the supplied snapshot uses a different setting-name template from training**, so ensure the correct saved checkpoint path is resolved before using evaluation-only mode.

## Reproducibility notes

- Data splits are chronological, and scalers are fitted only on the training partition.
- Unless otherwise stated, the representative experiments use multivariate-to-multivariate forecasting (`--features M`).
- Wavelet basis/level and patch configurations are fixed hyperparameters, not dynamically selected by AdaTM.
- Check CUDA, PyTorch, `mamba-ssm`, and the locally included wavelet implementation when reproducing results on new machines.
- The model's registered CLI name (`AdaTM`) is retained for compatibility with the supplied implementation.
- Some recorded horizon-specific runs include command typos or omitted CLI arguments; refer to successful run logs when validating exact paper results.

## Acknowledgements

This implementation includes components or conventions associated with [SST](https://arxiv.org/abs/2404.14757), [Mamba](https://github.com/state-spaces/mamba), [PatchTST](https://github.com/yuqinie98/PatchTST), and [Autoformer](https://github.com/thuml/Autoformer). We thank the authors of these open-source projects.

## Citation

If you use AdaTM in your research, please cite the associated paper, **“AdaTM: Adaptive Fusion of Transformer and Mamba for Time Series Forecasting.”** The final bibliographic/BibTeX record should be added here once the publication metadata is available.
