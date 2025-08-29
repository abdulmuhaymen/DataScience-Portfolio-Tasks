

# Random Forest Classifier: Scratch vs Scikit-learn
# 📌 Introduction

This project compares a Random Forest Classifier implemented from scratch with the optimized scikit-learn RandomForestClassifier, using a loan approval dataset.

The goal is to highlight both the conceptual understanding gained from a scratch implementation and the practical efficiency of scikit-learn’s version.

# 📂 Dataset

The dataset contains loan application records with the following features:

no_of_dependents

education

self_employed

income_annum

loan_amount

loan_term

cibil_score

residential_assets_value

commercial_assets_value

luxury_assets_value

bank_asset_value

Target variable:

loan_status → 1 (approved), 0 (not approved)

Preprocessing Steps:

Encoded categorical variables (education, self_employed, loan_status)

Split into 80% training and 20% testing

# ⚙️ Random Forest (Scratch Implementation)

The Random Forest was built manually with the following steps:

Bootstrapping – random sampling of subsets

Random Feature Selection – selecting a random subset of features at each split

Decision Tree Building – splitting using entropy/variance

Aggregation – majority voting for classification

Limitations:

Slower training (manual Python loops)

No optimization (no parallelization or pruning)

Manual handling of missing values and categorical encoding

# ⚡ Random Forest (Scikit-learn Implementation)

The scikit-learn implementation used:

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


Optimized and scalable

Wide range of hyperparameters

# 📊 Evaluation Metrics
	Scratch Implementation	

Accuracy--->84.5%

Precision--->85.2%

Recall--->84.5%	

F1-score--->83.9%	
# 📊 Evaluation Metrics
    Scikit-learn Implementation
Accuracy--->97.8%

Precision--->97.8%

Recall--->97.8%

F1-score--->97.8%
#
    ✅ Conclusion

The scratch implementation is valuable for understanding the mechanics of Random Forests but is slower and less accurate (~84.5%).

The scikit-learn implementation is highly optimized, efficient, and achieved ~98% accuracy.

For real-world applications, scikit-learn is the practical choice.

For learning and conceptual understanding, the scratch version is extremely useful.
