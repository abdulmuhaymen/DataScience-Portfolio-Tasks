🛒 Amazon Product Dataset - EDA

This project performs Exploratory Data Analysis (EDA) on an Amazon products dataset. The goal is to clean, preprocess, explore, and visualize the dataset to prepare it for further analysis or modeling.

📌 Features

Load and inspect dataset

Data cleaning & preprocessing

Outlier detection & removal

Encoding categorical variables

Exploratory visualizations

Model experimentation (Decision Tree & Random Forest)

🧹 Data Cleaning Techniques

The following preprocessing and cleaning steps were applied:

Cleaned numeric columns by removing special characters and converting to numeric

Removed missing values using dropna

Checked and removed duplicate rows

Outlier detection and removal using Z-score method

Dropped irrelevant columns (e.g., IDs, links, text-heavy fields)

Encoded categorical variables using LabelEncoder

📊 Visualizations

Various graphs and plots were created to better understand the dataset:

Histograms – Distribution of numerical features

Bar Plots – Comparison across categories

Count Plots – Frequency of categorical values

Scatter Plots – Relationship between features

Heatmap (Correlation Matrix) – Feature correlation insights

🤖 Models Used

The notebook also explores basic modeling techniques:

Decision Tree

Random Forest Classifier/Regressor

(Notebook structure suggests readiness for other sklearn models as well)