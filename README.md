# SparseBank

SparseBank is a machine-learning–assisted framework for reducing matched-filter template banks. By learning the object masses it allows to significantly reduce the number of templates while preserving detection sensitivity at significantly lower computational cost.

## Pipeline Overview

SparseBank consists of four steps:

### 1. Signal & Background Generation
Generate simulated BNS (Binary Neutron Star) signal and background strain data. Waveforms are injected into realistic noise realizations at a range of SNRs to produce labelled training, validation, and test sets in HDF5 format.

### 2. Mass Regression
Train a neural network based on the S4D (Structured State Space Diagonal) architecture to predict the masses and associated uncertainties of the two neutron stars directly from the strain time series. The model is trained using the datasets produced in Step 1.

### 3. Template Bank Filtering
Use the mass predictions and their uncertainties from Step 2 to prune a pre-existing `gstlal` template bank. Templates whose chirp mass falls outside the predicted range (plus a safety margin) are removed, producing a significantly smaller bank that still covers the relevant parameter space.

### 4. Matched Filtering
Run the full matched-filter pipeline (gstlal) on both signal and background data using the reduced template bank from Step 3. The lower template count directly translates to reduced computational cost while maintaining detection sensitivity.

## Installation

```bash
# Clone the repository
git clone https://github.com/chreissel/SparseBank.git
cd SparseBank

# Create and activate the conda environment
mamba env create -f env.yml
conda activate SparseBank
```
## Usage

Run the full pipeline:
```bash
python main.py --config config.yaml
```

Run individual steps:
```bash
python main.py --config config.yaml --steps 1        # generation only
python main.py --config config.yaml --steps 2,3,4    # train + filter + match
```

## Where Do We Stand

The overall pipeline structure is in place and all four steps are implemented end to end. The current focus is on hardening individual components before running a full integration test.

- [ ] Update data generation to ml4gw language (real background)
- [ ] Test training pipeline
- [ ] Add functionality to load pretrained model
- [ ] Test steps 3 and 4 of the pipeline
