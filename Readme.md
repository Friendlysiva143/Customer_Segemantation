# Customer Segmentation

A practical data science project focused on segmenting customers using unsupervised machine learning techniques and exploratory data analysis for better marketing strategies.

## Table of Contents
- Project Overview
- Dataset
- Features
- Requirement Document
- Approach & Methodology
- Files & Structure
- Usage
- Dependencies
- Results
- Deployment
- Author

---

## Project Overview

This repository implements a **customer segmentation** solution using clustering algorithms. The goal is to analyze customer data, group customers based on similar characteristics, and provide actionable insights for targeted marketing campaigns.

---

## Dataset

- **EDA_Data.csv**: Main dataset used for analysis.
- **marketing_campaign.xlsx**: Contains details about the marketing campaign.
- **EDA_df_clusters.csv**: Clustered data for further analysis.

---

## Features

- **Features.txt**: List of features included in the dataset
- Multiple features explored for segmentation, such as demographics, purchase history, and campaign response.

---

## Requirement Document

- **Requirement document.docx**: Outlines project objectives, business requirements, and deliverables.

---

## Approach & Methodology

- **Jupyter Notebook**: Step-by-step code for preprocessing, EDA, and clustering (`Market_Customer_segmentation.ipynb`)
- Data Preprocessing: Imputation, scaling, and feature reduction (PCA)
- Clustering: Algorithm selection, tuning, and interpretation
- Post-analysis: Cluster profiling and marketing recommendations

---

## Files & Structure

- `EDA_Data.csv` — Raw data for EDA
- `EDA_df_clusters.csv` — Data with cluster assignments
- `Features.txt` — Feature list
- `Market_Customer_segmentation.ipynb` — Main notebook for analysis
- `Requirement document.docx` — Business requirements
- `final_model.pkl` — Saved clustering model
- `pca.pkl` — Principal Component Analysis model
- `scaler.pkl` — Scaler used during preprocessing
- `problem_statement.txt` — Project problem statement
- `requirements.txt` — Required packages (see below)
- `streamlit.py` — Streamlit app for visualization and deployment

---

## Usage

1. Clone this repository
2. Install packages from `requirements.txt`
3. Run the Jupyter notebook for analysis
4. Launch `streamlit.py` for interactive cluster visualization (requires Streamlit)

---

## Dependencies

Install packages using:
pip install -r requirements.txt


---

## Results

- Segmented customer groups
- Actionable marketing insights based on clusters
- Interactive dashboard for cluster visualization

---

## Deployment

- Streamlit app (`streamlit.py`) for easy sharing of insights and interactive exploration.

---

## Author

- Siva Prasad (Friendlysiva143) — Aspiring Data Scientist with interests in clustering, EDA, and predictive modeling.


