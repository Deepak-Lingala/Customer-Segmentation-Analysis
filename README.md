# Advanced E-Commerce Customer Segmentation Analysis

## Overview
This project showcases a senior-level approach to unsupervised machine learning by applying K-Means Clustering on a massively scaled, 10-dimensional synthetic dataset of 500,000 e-commerce customers. Moving beyond basic RFM (Recency, Frequency, Monetary) analysis, this model integrates complex behavioral web metrics to derive highly nuanced, actionable business personas.

Built from scratch, this repository serves as a portfolio piece highlighting end-to-end data generation, preprocessing, PCA dimensionality reduction, and cohort analysis.

## Business Problem
While standard RFM models identify who spends the most, they fail to capture the *cost* of servicing those customers or the *context* of their buying habits. A shopper who buys $1,000 worth of merchandise but returns 60% of it and logs 4 support calls is inherently different from a frictionless VIP. The goal of this analysis is to structurally identify these deep behavioral clusters so the business can intervene strategically.

## Methodology
1. **Algorithmic Data Generation**: Built a fully vectorized Python data generator using NumPy to synthesize 500,000 rows across 10 features without memory overflow:
   - `Recency`, `Frequency`, `Monetary`, `Tenure`
   - `Return_Rate`, `Discount_Usage`, `Satisfaction_Score`
   - `Avg_Time_Site_Min`, `Support_Calls`
2. **Preprocessing & Feature Engineering**: 
   - Applied log transformations to heavily right-skewed variables (Frequency, Monetary, Support Calls) to normalize distributions.
   - Standardized the multi-dimensional array using `StandardScaler`.
3. **Machine Learning (Clustering)**: 
   - Utilized the **Elbow Method** and **Silhouette Scoring** (on representative samples) to optimize the hyperparameter $k$, settling on $k=5$ distinct clusters.
   - Built the final K-Means model to output cluster classifications.
4. **Dimensionality Reduction**: Visualized the 9 scaled feature dimensions on a clear 2D plane using Principal Component Analysis (PCA).

## Key Findings & Personas
By analyzing the centroids of the 5 clusters across all dimensions, we identified specific behavioral taxonomies:
1. **Zero-Friction VIPs**: High spenders with zero returns and perfect satisfaction.
2. **High-Value but Churn-Risk**: Heavy spenders burdened by high return rates, significant support call volume, and poor satisfaction scores.
3. **Discount-Driven Shoppers**: High transaction volume exclusively utilizing discount codes to achieve lower monetary totals.
4. **New/Occasional Buyers**: Low tenure shoppers still building brand affinity.
5. **Dormant/Lost**: Long-standing accounts that have ceased purchasing.

## Real-World Marketing Impact
By integrating satisfaction and return rates into the clustering:
- Customer Service teams can prioritize proactive outreach to the **High-Value Churn-Risks**.
- Margin bleed can be stopped by shifting **Discount-Driven Shoppers** to volume-based tier promotions instead of flat % discounts.
- Marketing budgets are preserved by suppressing **Dormant/Lost** customers from expensive direct-mail campaigns in favor of cheap retargeting.
*(See `marketing_strategy.md` for the full deployment strategy.)*

## How to Run
1. Ensure Python 3.11 is installed.
2. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate the 10-feature dataset (vectorized runtime < 10 seconds):
   ```bash
   python generate_data.py
   ```
4. Directly Open the Pre-Executed Jupyter Notebook:
   ```bash
   jupyter notebook segmentation.ipynb
   ```