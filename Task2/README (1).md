
---
# 🌳 Decision Tree (ID3) From Scratch — Entropy & Information Gain

This project implements a **Decision Tree classifier from scratch** (no scikit-learn tree APIs). It walks through the **mathematics** (Entropy & Information Gain), the **algorithm** (ID3), and a **clean Python implementation**. The notebook uses a small “**Play Tennis**” dataset (`tennis.csv`) to illustrate how the root and subsequent nodes are chosen.

> Why this project?
> - Understand exactly how a decision tree picks splits.
> - See Entropy & Information Gain computed step by step.
> - Learn how recursion builds the tree and how prediction traverses it.

---

## ✨ Key Features

- Pure-Python implementation of **Entropy** and **Information Gain**  
- **ID3** algorithm to build a tree recursively  
- **Root node selection** explained & computed from the data  
- A simple **predict** function that walks the learned tree  
- Clear, readable code with minimal dependencies (`numpy`, `pandas`)

---

## 📁 Project Structure


## 🧮 Theory (Concise)

### Entropy (impurity of a label set)
For a dataset \(S\) with class proportions \(p_i\):
\[
Entropy(S) = - \sum_i p_i \log_2(p_i)
\]
- 0 = pure (all one class)  
- 1 = maximally impure (balanced binary classes)

### Information Gain (how much uncertainty a split removes)
For feature \(A\) with values \(v\):
\[
IG(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \cdot Entropy(S_v)
\]
Pick the feature with the **highest IG** → becomes the **split** (root if first).

---

## 🔁 ID3 Algorithm (High Level)

1. **Base cases**
   - If all labels are the same → return that label (leaf).
   - If no features remain → return the **majority class** (leaf).

2. **Recursive step**
   - Compute **Information Gain** for all remaining features.
   - Split on the feature with **max IG**.
   - Recurse on each subset; remove the used feature from the list.

---

## 🧪 Example: Root Selection (Play Tennis)

> If you use the classic 14-row “Play Tennis” dataset (9 “Yes”, 5 “No”), then:
- Overall entropy  
  \[
  Entropy(S) = -\tfrac{9}{14}\log_2\tfrac{9}{14} - \tfrac{5}{14}\log_2\tfrac{5}{14} \approx 0.940
  \]
- Typical information gains (rounded):
  - \(IG(S,\text{Outlook}) \approx 0.246\) → **best**
  - \(IG(S,\text{Humidity}) \approx 0.152\)
  - \(IG(S,\text{Wind}) \approx 0.048\)
  - \(IG(S,\text{Temperature}) \approx 0.029\)

So **Outlook** becomes the **root node** (your code computes this from the CSV).  
> If your `tennis.csv` differs, numbers will adjust automatically; logic stays the same.

---

## 🧰 Environment & Setup

```bash
pip install pandas numpy
# optional: jupyter
pip install notebook


▶️ End-to-End Flow

Load dataset (tennis.csv).

Extract features & target.

Build tree using id3.

Predict outcomes using predict.

🧩 Practical Notes

Categorical features: ID3 works naturally with them.

Unseen categories: prediction may return None; can be replaced by majority class.

Continuous features: extend with threshold-based splits.

Pruning: can be added to avoid overfitting.

✅ What You’ll Learn

How Entropy measures impurity.

How Information Gain selects the best feature.

How recursion builds a tree structure.

How to predict by walking down the branches.