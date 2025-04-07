# Data Matching Project

This project implements a machine learning-based approach to match Points of Interest (POIs) across different data sources. The system uses a combination of feature engineering, candidate generation, and XGBoost classification to identify matching businesses.

## Overview

The data matching pipeline consists of several key components:

1. **Data Extraction**: Loading data from multiple sources
2. **Data Transformation**: Preprocessing and feature engineering
3. **Candidate Generation**: Efficiently identifying potential matches
4. **Model Training**: Training an XGBoost classifier on labeled data
5. **Prediction**: Scoring and classifying candidate pairs

## Project Structure

```
data_matching_project/
├── data/                  # Input data files
│   ├── data_source_1.csv  # First dataset
│   └── data_source_2.csv  # Second dataset
├── output/                # Output files
│   ├── matches.csv        # Matched pairs
│   └── model/             # Trained models
├── pipeline/              # Pipeline code
│   ├── conflation.py      # Main matching pipeline
│   ├── data_gen.py        # Data generation utilities
│   ├── feature_engineering.py  # Feature extraction
│   ├── feature_gen.py     # Feature generation script
│   ├── model_gen.py       # Model training script
│   └── utils.py           # Utility functions
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/data_matching_project.git
cd data_matching_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

### 1. Data Preparation

Place your input data files in the `data/` directory:
- `data_source_1.csv`: First dataset
- `data_source_2.csv`: Second dataset

### 2. Generate Training Data

To generate training data for model development:

```bash
python pipeline/data_gen.py --ds1 data/data_source_1.csv --ds2 data/data_source_2.csv --output output/training_data --max-negative 1000 --include-low-distance
```

Parameters:
- `--ds1`: Path to the first dataset
- `--ds2`: Path to the second dataset
- `--output`: Output directory for training data
- `--max-negative`: Maximum number of negative pairs to generate
- `--include-low-distance`: Include low distance negative pairs

### 3. Generate Features

To extract features from the training data:

```bash
python pipeline/feature_gen.py --input output/training_data/train_data.csv --output output/features/train_features.csv
```

Parameters:
- `--input`: Path to the input data file
- `--output`: Path to save the extracted features
- `--embedding-model`: Path to the embedding model (optional)

### 4. Train the Model

To train a model on the extracted features:

```bash
python pipeline/model_gen.py --train output/features/train_features.csv --val output/features/val_features.csv --output output/model/xgboost_special_negatives --model-type xgboost --tune
```

Parameters:
- `--train`: Path to the training features CSV file
- `--val`: Path to the validation features CSV file
- `--output`: Directory to save the model and evaluation results
- `--model-type`: Type of model to train (choices: "logistic", "xgboost", "dnn", default: "xgboost")
- `--tune`: Perform hyperparameter tuning (flag)

#### XGBoost Parameters
- `--n-estimators`: Number of trees (default: 100)
- `--max-depth`: Maximum depth of trees (default: 3)
- `--learning-rate`: Learning rate (default: 0.1)

### 5. Run the Matching Pipeline

To run the full matching pipeline:

```bash
python pipeline/conflation.py --input1 data/data_source_1.csv --input2 data/data_source_2.csv --output output/matches.csv --model output/model/xgboost_special_negatives/xgboost_model.pkl --scaler output/model/xgboost_special_negatives/xgboost_scaler.pkl --threshold 0.5
```

Parameters:
- `--input1`: Path to the first input dataset
- `--input2`: Path to the second input dataset
- `--output`: Path to save the output matches
- `--model`: Path to the trained model (optional)
- `--scaler`: Path to the feature scaler (optional)
- `--threshold`: Probability threshold for match classification (default: 0.5)

## Experiments

### Experiment 1: Basic Matching

The initial experiment used a simple approach with:
- Embedding cosine similarity for name, address and categories
- Geographic distance calculation
- XGBoost classifier with default parameters

Results:
- Accuracy: 0.92
- Precision: 0.94
- Recall: 0.85
- F1 Score: 0.89

### Experiment 2: Improved Feature Engineering

This experiment enhanced the feature engineering:
- Added n-gram hashing for name matching
- Implemented geohash for spatial indexing
- Added postcode extraction and matching
- Improved category similarity using embedding-based approach

Results:
- Accuracy: 0.93
- Precision: 0.95
- Recall: 0.86
- F1 Score: 0.90

### Experiment 3: Special Negative Examples

This experiment focused on improving the model's ability to distinguish between similar businesses at different locations:
- Added special negative examples with similar names but different locations
- Balanced the training data to include more challenging cases
- Adjusted model parameters to handle these cases better

Results:
- Accuracy: 0.95
- Precision: 0.97
- Recall: 0.85
- F1 Score: 0.91

### Experiment 4: Threshold Optimization

This experiment explored different probability thresholds:
- Tested thresholds from 0.3 to 0.7 in increments of 0.05
- Evaluated the impact on precision, recall, and F1 score
- Selected the optimal threshold based on the F1 score

Results:
- Best threshold: 0.5
- Accuracy: 0.95
- Precision: 0.97
- Recall: 0.85
- F1 Score: 0.91

### Experiment 5: Model Comparison

This experiment compared different model architectures:
- XGBoost
- Random Forest
- LightGBM
- Neural Network

Results:
- XGBoost performed best with:
  - Accuracy: 0.95
  - Precision: 0.97
  - Recall: 0.85
  - F1 Score: 0.91
- Neural Network required more data to perform well

## Model Performance

The final model achieves:
- **Accuracy**: 0.9514 (95.14% of predictions are correct)
- **Precision**: 0.9722 (97.22% of predicted matches are actual matches)
- **Recall**: 0.8537 (85.37% of actual matches are correctly identified)
- **F1 Score**: 0.9091 (harmonic mean of precision and recall)
- **ROC AUC**: 0.9873 (area under the ROC curve)

## Scaling Considerations

For handling millions of POIs, consider:
1. Using geospatial indexing (R-tree, QuadTree)
2. Implementing distributed processing (MapReduce, Spark)
3. Optimizing feature computation with caching
4. Designing an incremental processing system
5. Using model compression techniques