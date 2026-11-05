# Molecular Activity Prediction

This repository contains the code, analysis, and presentation for predicting molecular activity using Machine Learning and Cheminformatics techniques in R.

## Project Overview
The goal of this project is to analyze molecular fingerprints, perform rigorous feature selection, and build robust classification models to predict the activity levels of chemical compounds against a specific enzyme.

* **Dataset Size:** 2,594 unique molecules (after data cleaning).
* **Features:** 169 binary variables representing molecular MACCS keys/substructures.
* **Target Variable (`Labels`):** Three activity classes:
  * `0`: Not active (58.1%)
  * `1`: Weakly active (26.0%)
  * `2`: Strongly active (15.9%)

---

## Repository Structure
- `Predicting the activity of a molecule.R`: The complete R script containing data preprocessing, exploratory data analysis, PCA, clustering, feature selection (LASSO, Boruta, RFE, Mutual Information), classification model training (SVM, Random Forest, Multinomial Logistic Regression), and Association Rules mining.
- `Final_Molecular_Activity_Presentation.pptx`: The summary presentation slides detailing the pipeline and final results.

---

## Methodology & Pipeline

1. **Data Preprocessing & Cleaning:**
   - Removed index columns, checked for duplicates, and handled missing values.
   - Filtered out variables with zero variance.
   - Handled class imbalance using strategic class weighting during model training.

2. **Exploratory Data Analysis & Dimensionality Reduction:**
   - Analyzed target variable distribution and feature variances/means.
   - Performed **PCA**, demonstrating that low-dimensional projections exhibit heavy class overlap (requiring ~70 components to explain 95% of variance).
   - Applied **One-Way ANOVA** across features to identify statistically significant differences among the activity groups.

3. **Clustering Analysis:**
   - Compared **Hierarchical Clustering** (using Euclidean distance on PCA scores vs. Jaccard distance on binary data) and **K-Means**.
   - Demonstrated that Jaccard distance handles binary fingerprint data more effectively than Euclidean distance.

4. **Feature Selection:**
   - Implemented and compared four distinct feature selection techniques: **LASSO**, **Boruta**, **Recursive Feature Elimination (RFE)**, and **Mutual Information**.
   - Combined top-ranking features dynamically to evaluate subsets ranging from 20 to 120 features. Key recurring features identified include `NS`, `OCO`, and `5 M ring`.

5. **Classification & Model Evaluation:**
   - Trained and evaluated **Support Vector Machines (SVM)**, **Random Forest (Ranger)**, and **Multinomial Logistic Regression (MLR)** using 70/30 train-test splits.
   - Evaluated models using Accuracy, Kappa, Balanced Accuracy, and Macro F1-Score.

---

## Key Results & Best Variables

* **Best Performing Model:** **Random Forest (ranger)** using **90 dynamic mutual features** achieved the highest performance:
  * **Accuracy:** 72.6%
  * **Macro F1-Score:** 0.66
  * **Kappa:** 0.50
* **Top Predictive Substructures:** Features like `NS`, `OCO`, `5 M ring`, `A!N$A`, and `CSN` consistently appeared as the most important discriminators across multiple feature selection algorithms.
* **Association Rules Mining (`apriori`):** Extracted high-confidence rules indicating that specific substructure combinations (e.g., containing `CSN`, `SAN`, and `NH2`) strongly associate with Class 2 (strongly active molecules) with a lift of up to 5.59.

---

## Requirements & Libraries
The analysis was conducted in R using the following key packages:
`readxl`, `caret`, `corrplot`, `e1071`, `randomForest`, `ranger`, `glmnet`, `Boruta`, `FSelectorRcpp`, `arules`, `arulesViz`, `factoextra`, `cluster`.
