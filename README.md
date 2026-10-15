# 🛰️ Multi-Temporal Land Use/Land Cover Classification with BigEarthNet-v2

**A multi-label remote-sensing project exploring temporal information in Sentinel-2 multispectral imagery.**

## Overview

Land Use/Land Cover (LULC) classification is an important task in remote sensing, supporting environmental monitoring, land management, and geospatial analysis.

This project investigates whether **multiple observations of the same geographic location, acquired at different dates, can improve multi-label land-cover classification compared with a single observation**.

Using the BigEarthNet-v2 dataset, the project builds a temporal learning pipeline that combines multispectral image preprocessing, pretrained feature extraction, temporal sequence modelling, and comparative evaluation.

The repository also includes a Streamlit research dashboard for exploring temporal imagery, learned feature representations, and experimental results.

## Research Question

> Does incorporating temporal information from four Sentinel-2 observations improve multi-label LULC classification compared with using a single observation?

To investigate this question, the project compares single-date and multi-date representations while exploring different strategies for aggregating or learning from temporal features.

## Key Features

- **Multispectral preprocessing:** Preparation of six Sentinel-2 spectral bands at a common spatial resolution.
- **Temporal sequence construction:** Four observations from distinct acquisition dates for each eligible spatial location.
- **Multi-label classification:** Prediction of 19 land-cover classes using multi-hot target vectors.
- **Pretrained feature extraction:** Use of ResNet50 to extract 2,048-dimensional feature vectors from individual observations.
- **Temporal modelling:** Experiments with feature averaging, gated recurrent units (GRUs), and attention-based approaches.
- **Comparative evaluation:** Single-date and four-date experiments using validation and test metrics.
- **Interactive dashboard:** Streamlit-based exploration of temporal imagery, feature activations, and model comparisons.
- **Reproducible data preparation:** Scripts for metadata analysis, split verification, manifest construction, normalization, and experimental workflows.

## Dataset

The project uses **BigEarthNet-v2**, a large-scale remote-sensing benchmark containing Sentinel-2 satellite observations with multi-label land-cover annotations.

| Property | Description |
|---|---|
| Dataset | BigEarthNet-v2 |
| Satellite | Sentinel-2 |
| Task | Multi-label LULC classification |
| Number of classes | 19 |
| Spectral bands used | 6 |
| Observations per temporal sequence | 4 |
| Spatial resolution of prepared inputs | 120 × 120 pixels |

### Dataset Cohort

The four-date cohort constructed during data preparation contains the following locations:

| Split | Spatial locations | Observations |
|---|---:|---:|
| Training | 14,812 | 59,248 |
| Validation | 7,703 | 30,812 |
| Test | 7,496 | 29,984 |
| **Total** | **30,011** | **120,044** |

Each eligible location has four observations from distinct acquisition dates. The original dataset split assignments are retained to prevent spatial locations from crossing the training, validation, and test boundaries.

These figures describe the prepared four-date cohort, not the entire BigEarthNet-v2 dataset.

### Multispectral Representation

Each prepared observation contains six spectral channels:

| Channel | Sentinel-2 band | Description |
|---:|---|---|
| 0 | B02 | Blue |
| 1 | B03 | Green |
| 2 | B04 | Red |
| 3 | B08 | Near-infrared |
| 4 | B11 | Short-wave infrared 1 |
| 5 | B12 | Short-wave infrared 2 |

The RGB, NIR, and SWIR fields are decoded and aligned to a common 120 × 120 grid. The SWIR bands, originally represented at 60 × 60 pixels in the prepared data, are resized using bilinear interpolation.

A single observation has the tensor shape:

```text
(6, 120, 120)
```

A four-date sequence has the tensor shape:

```text
(4, 6, 120, 120)
```

The first dimension represents acquisition dates, while the second represents spectral channels.

### Multi-Label Targets

Each spatial location is represented by a 19-dimensional multi-hot target vector:

- `1` indicates that a land-cover class is present.
- `0` indicates that a land-cover class is not assigned to that location.

The dataset labels describe the land-cover categories associated with a location. They are not date-specific change annotations, so the task should not be interpreted as direct land-cover change detection.

## Methodology

### 1. Data Preparation

The preprocessing pipeline includes:

- BigEarthNet-v2 metadata analysis.
- Verification of temporal observations and original dataset splits.
- Identification of locations with four distinct acquisition dates.
- Construction of temporal manifests and spatial-location mappings.
- Decoding and alignment of six-band multispectral observations.
- Validation of label consistency across temporal sequences.
- Creation of multi-hot target vectors.
- Computation of normalization statistics from training observations only.

### 2. Feature Extraction

A pretrained ResNet50 is used to transform each prepared observation into a learned feature representation.

Each observation produces a feature vector with 2,048 dimensions. A four-date sequence therefore has the representation:

```text
(4, 2048)
```

These features provide a compact representation for studying temporal modelling without repeatedly processing the original multispectral imagery during every classifier experiment.

### 3. Temporal Modelling

The experimental workflows investigate different ways to use information across the four observations:

- **Single-date baseline:** Classification using one observation's feature representation.
- **Temporal feature averaging:** Aggregation of features across four dates.
- **GRU-based classification:** Learning sequential representations with a recurrent neural network.
- **Attention-based modelling:** Exploring mechanisms for weighting or combining temporal information.

The experiments are intended to distinguish the benefits of additional observations from the effects of changing the model architecture.

### 4. Evaluation

The project uses multi-label classification metrics, including:

- **Macro-F1:** Gives equal weight to the F1 score of each class.
- **Micro-F1:** Aggregates decisions across classes before calculating the F1 score.
- **Mean Average Precision (mAP):** Evaluates ranking quality across classes.
- **Loss:** Tracks the model's optimisation objective.

Validation data should guide model selection and tuning. The held-out test split should be reserved for final evaluation rather than repeated experimentation.

## Research Dashboard

The Streamlit dashboard provides an interactive interface for examining the project.

Its intended capabilities include:

- Comparing single-date and multi-date experimental results.
- Exploring temporal observations for a selected location.
- Inspecting multispectral imagery and temporal sequences.
- Visualising the four-date, 2,048-dimensional feature representation as a heatmap.
- Understanding how image observations are transformed into learned features.

The dashboard is designed as a research companion to the modelling pipeline. It visualises saved experimental outputs and available local data rather than replacing the training and evaluation workflows.

## Repository Structure

```text
BigEarthNet-Temporal-LULC/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── data/
├── docs/
│   └── data_processing.md
├── scripts/
│   ├── 15_build_full_temporal_manifest.py
│   ├── 16_compute_band_statistics.py
│   ├── 17_build_pytorch_dataset.py
│   ├── 18_temporal_resnet_model.py
│   ├── 25_compare_single_vs_four_dates.py
│   ├── 26_improved_single_vs_four_dates.py
│   ├── 27_full_data_single_vs_four_dates.py
│   ├── 29_extract_temporal_features.py
│   ├── 30_train_gru_temporal_classifier.py
│   ├── 31_train_small_gru_temporal_classifier.py
│   ├── 32_train_temporal_attention.py
│   └── 33_controlled_average_vs_attention.py
├── temporal_data/
│   └── [generated manifests, metrics and local experiment data]
└── scripts/archive/
    ├── README.md
    └── [archived data preparation and diagnostic scripts]
```

This is a representative structure of the main project files. Generated datasets, feature caches, checkpoints, and other local experiment artifacts may be excluded from version control.

## Installation

### Prerequisites

- Python 3.10 or a compatible Python version supported by the installed dependencies.
- Git.
- Access to the BigEarthNet-v2 data required for the experiments.
- Sufficient memory and storage for dataset processing and model execution.

### 1. Clone the repository

```bash
git clone https://github.com/sunandahs05/Multi-Label-Land-Use-Land-Cover-Classification-on-Sentinel-2-Multispectral-Imagery.git
cd Multi-Label-Land-Use-Land-Cover-Classification-on-Sentinel-2-Multispectral-Imagery
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Install the dashboard dependencies if they are not already included in the environment:

```bash
python -m pip install streamlit plotly
```

PyTorch and the remaining dependencies must be compatible with your Python version and hardware.

### 3. Launch the dashboard

From the project root, run:

```bash
python -m streamlit run app.py
```

The dashboard depends on the local data and experiment artifacts available to it. Some visualisations may be unavailable if their required files have not been generated or are excluded from the repository.

## Reproducibility and Data Management

The repository separates source code from generated artifacts to keep the project manageable.

- Dataset downloads and raw data should remain outside version control.
- Large feature caches and model checkpoints are generated locally.
- Small sample files may be used for demonstrations where appropriate.
- Archived scripts preserve earlier data preparation and diagnostic workflows.
- Training, validation, and test splits should remain separate throughout experimentation.
- Normalisation statistics must be estimated using training data only.
- Final model comparisons should document their data cohort, random seed, training configuration, and evaluation split.

Refer to [`docs/data_processing.md`](docs/data_processing.md) for additional data-processing documentation.

## Current Research Status

The repository contains data preparation workflows, temporal feature extraction, single-date and multi-date comparison experiments, and GRU and attention-based modelling scripts.

The Streamlit dashboard is being developed to make the pipeline and its experimental outputs easier to inspect.

The relative performance of the approaches should be reported using the corresponding verified evaluation results. The presence of an experiment script alone does not imply that its results have been fully validated or that one model is definitively superior.

## Limitations

- The prepared cohort is restricted to locations with four eligible observations.
- Resizing spectral bands to a common grid changes their original spatial sampling.
- Pretrained feature extraction may not fully exploit the characteristics of multispectral imagery.
- The 19-class task is multi-label classification, not pixel-wise semantic segmentation.
- Temporal observations with static location-level labels do not directly measure land-cover change.
- Model comparisons depend on consistent data splits, evaluation protocols, and training configurations.

## Future Work

Potential extensions include:

- More rigorous controlled comparisons of single-date and temporal models.
- Evaluation of temporal aggregation and attention strategies.
- Per-class performance analysis and investigation of class imbalance.
- Improved temporal visualisation and interpretability.
- Stronger reproducibility through configuration files and experiment tracking.
- Further evaluation of spatial generalisation and robustness.

## Acknowledgements

This project uses the BigEarthNet-v2 remote-sensing dataset and builds on the open-source Python, PyTorch, and Streamlit ecosystems.

Please consult the dataset's official documentation and licence terms before redistributing any data or derived artifacts.

---

*This repository is an ongoing applied machine-learning project focused on temporal remote sensing, multispectral feature learning, and multi-label land-cover classification.*