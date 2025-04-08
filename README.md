# Improving Prediction Region Accuracy in Marine Animal Movement with Temporal Fusion Transformer

This repository contains the code for "Improving Prediction Region Accuracy in Marine Animal Movement with Temporal Fusion Transformer".

Our implementation relies on the [pytorch_forecasting](https://github.com/sktime/pytorch-forecasting) package.

## Getting Started

Replace `dataset.csv` and `metadata.csv` under the `data` directory with your specific data.

### dataset.csv

Contains the time series for each animal. The required columns are:

- `LATITUDE`: The latitude coordinate of the animal's position.
- `LONGITUDE`: The longitude coordinate of the animal's position.
- `DATE_TIME`: The date and time of the recorded position.
- `ID`: A unique identifier for the time series corresponding to each animal.

#### Format of `dataset.csv`

```plaintext
LATITUDE, LONGITUDE, DATE_TIME, ID
-34.9285, 138.6007, 2020-01-01 12:00:00, A123
-33.8688, 138.8093, 2020-01-01 13:00:00, A123
...
```

### metadata.csv

Contains static information about each animal. The required columns are:

- `ID`: The unique identifier corresponding to each animal (matching `ID` in `dataset.csv`).
- `Species`: The species of the animal.

Other static data (e.g. `Weight`, `Sex`, etc.) can be passed to the model by through the `static_categoricals` and `static_reals` arguments.

#### Format of `metadata.csv`

```plaintext
ID, Species, Sex
A123, Southern elephant seal, M
B456, Southern elephant seal, F
...
```

## Important Notes

- Ensure that the `ID` column is consistent across both `dataset.csv` and `metadata.csv` files.
- The date and time format in `DATE_TIME` should be consistent and in a standard format (e.g., YYYY-MM-DD HH:MM:SS).

## Usage

Run the scripts under the `forecasting/scripts` directory for time-demanding computations. Use the `forecast.ipynb` notebook for interactive computation and plotting. To download complementary environmental variables from the [Copernicus ERA5 dataset](https://cds.climate.copernicus.eu/doi/10.24381/cds.adbb2d47) adjust the scripts within `ERA5-Land-data-analysis-main`.

## Installation

Clone the conda environment

```bash
conda env create -f environment.yml
```
and activate it

```bash
conda activate animaltorch
```
