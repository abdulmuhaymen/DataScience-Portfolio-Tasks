# 🛒 Smart Shopping Cart - Market Basket Analysis

A machine learning-powered recommendation system that provides real-time product suggestions based on shopping cart contents using Apriori and TF-IDF algorithms.



## 🎯 Overview

This project implements a smart shopping cart system that analyzes historical transaction data to provide intelligent product recommendations. When users add items to their cart, the system instantly suggests complementary products that other customers frequently purchase together.

### Key Capabilities
- **Real-time Recommendations**: Get instant product suggestions as you shop
- **Dual Algorithm Support**: Choose between Apriori and TF-IDF approaches
- **Interactive Shopping Cart**: Add/remove items with visual feedback
- **Confidence Scoring**: See recommendation strength with visual indicators
- **Export Functionality**: Save analysis results for further use

## ✨ Features

### 🛍️ Shopping Experience
- **Interactive Cart Management**: Easy add/remove items interface
- **Real-time Suggestions**: Instant recommendations based on cart contents
- **Visual Confidence Indicators**: Progress bars showing recommendation strength
- **Algorithm Transparency**: See which algorithm generated each recommendation

### 🤖 Machine Learning
- **Apriori Algorithm**: Classical association rule mining approach
- **TF-IDF Method**: Text-based similarity for product associations
- **Configurable Parameters**: Adjust support and confidence thresholds
- **Performance Metrics**: Support, confidence, and lift calculations

### 📊 Analytics
- **Cart Analytics**: Track items, recommendations, and confidence scores
- **Export Options**: Download results as CSV or pickle files
- **Data Visualization**: Interactive charts and metrics dashboard

## 🔬 Algorithms

### Apriori Algorithm
- **Purpose**: Finds frequent itemsets and generates association rules
- **Strengths**: Well-established, interpretable results
- **Best For**: Traditional market basket analysis with clear item relationships

### TF-IDF Approach
- **Purpose**: Uses text similarity to find product associations
- **Strengths**: Handles variations in product names, scalable
- **Best For**: Large catalogs with diverse product descriptions



### Step 1: Data Upload
- Upload a CSV or Excel file with transaction data
- Each row represents one transaction
- Each column represents an item position in the transaction

### Step 2: Algorithm Configuration
- Choose between **Apriori** or **TF-IDF** algorithm
- Adjust **Minimum Support** (0.005-0.3): How often items appear together
- Set **Minimum Confidence** (0.1-0.9): Recommendation strength threshold

### Step 3: Model Training
- Click "Initialize [Algorithm] Recommendations"
- Wait for model training to complete
- System will display the number of recommendation patterns found

### Step 4: Shopping & Recommendations
- Add items to cart using the dropdown selector
- View real-time recommendations with confidence scores
- Click ➕ to add recommended items to your cart
- Remove items using the ❌ button


#### Apriori Settings
- **Min Support**: 0.005-0.3 (default: 0.02)
  - Lower values find more patterns but increase computation time
  - Higher values focus on strongest associations only

- **Min Confidence**: 0.1-0.9 (default: 0.4)
  - Lower values show more recommendations with lower confidence
  - Higher values show fewer but stronger recommendations

#### TF-IDF Settings
- **Min Support**: 0.005-0.3 (default: 0.02)
  - Filters out rare item combinations
  
- **Min Confidence**: 0.1-0.9 (default: 0.4)
  - Controls recommendation threshold

### Performance Tuning
- **Large Datasets**: Increase min_support to reduce computation time
- **More Recommendations**: Decrease min_confidence threshold
- **Stronger Patterns**: Increase both support and confidence thresholds

