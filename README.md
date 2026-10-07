# Self-Quantified Fitness Tracker

This project is a data science workflow for analyzing wearable motion sensor data and building a self-quantified fitness tracking system. It uses accelerometer and gyroscope signals from MetaMotion devices to process raw sensor data, engineer meaningful features, and explore movement patterns such as exercise repetitions and activity categories.

## Project Overview

The repository demonstrates a complete pipeline for:
- loading raw CSV sensor files from wearable devices
- combining accelerometer and gyroscope data into a unified dataset
- resampling and cleaning time-series data
- extracting features such as PCA components, signal magnitude, temporal statistics, and frequency-domain characteristics
- visualizing and clustering movement patterns for analysis

## Dataset

The project uses MetaMotion sensor recordings that include:
- accelerometer measurements
- gyroscope measurements
- participant, label, category, and set metadata

These data files are stored in the data/raw/MetaMotion folder and are processed into interim datasets for further analysis.

## Workflow

The repository is organized around a typical machine learning and signal-processing pipeline:

```text
Raw Sensor CSVs
   ↓
Data Loading & Preprocessing
   ↓
Feature Engineering
   ↓
Temporal / Frequency Abstraction
   ↓
Modeling & Clustering
   ↓
Visualization & Insights
```

The main stages are:
1. Data loading and preprocessing in the data module
2. Feature engineering in the features module
3. Modeling and prediction logic in the models module
4. Visualization and exploratory analysis in the visualization module

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib

## Project Structure

- src/data: data loading and dataset construction
- src/features: signal processing, temporal abstraction, frequency features, and repetition analysis
- src/models: training and prediction logic
- src/visualization: plotting and exploratory visualization

## Getting Started

To get started with the project:

```bash
pip install -r requirements.txt
python src/data/make_dataset.py
```

This project can be extended to support exercise classification, repetition counting, fitness activity monitoring, and personalized health analytics.
