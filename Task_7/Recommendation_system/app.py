import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page config
st.set_page_config(
    page_title="Smart Shopping Cart - Product Recommendations",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .cart-item {
        background-color: #e8f4fd;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation {
        background-color: #f0f9ff;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #10b981;
        margin: 0.8rem 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .recommendation:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.15);
    }
    .confidence-bar {
        width: 100%;
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981, #059669);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    .empty-cart {
        text-align: center;
        padding: 2rem;
        color: #666;
        background-color: #f9f9f9;
        border-radius: 10px;
        border: 2px dashed #ddd;
    }
    .algorithm-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# APRIORI ALGORITHM FUNCTIONS
# ========================

def load_transactions_from_df(df):
    """Convert dataframe to transaction format"""
    transactions = []
    for i in range(len(df)):
        transaction = set()
        for col in df.columns:
            item = str(df.iloc[i][col])
            if item != "nan" and item.strip() != "" and item.lower() != "none":
                transaction.add(item.strip())
        if transaction:
            transactions.append(transaction)
    return transactions

def get_support(transactions, itemset):
    """Calculate support for an itemset"""
    count = sum(1 for t in transactions if itemset.issubset(t))
    return count / len(transactions) if transactions else 0

def get_frequent_itemsets(transactions, candidates, min_support):
    """Get frequent itemsets from candidates"""
    frequent = []
    support_data = {}
    for itemset in candidates:
        sup = get_support(transactions, itemset)
        if sup >= min_support:
            frequent.append(itemset)
            support_data[frozenset(itemset)] = sup
    return frequent, support_data

def generate_candidates(frequent_itemsets, k):
    """Generate candidate itemsets of size k"""
    candidates = []
    n = len(frequent_itemsets)
    for i in range(n):
        for j in range(i+1, n):
            l1 = list(frequent_itemsets[i])
            l2 = list(frequent_itemsets[j])
            l1.sort(); l2.sort()
            if l1[:k-2] == l2[:k-2]:
                candidate = frequent_itemsets[i] | frequent_itemsets[j]
                if candidate not in candidates:
                    candidates.append(candidate)
    return candidates

def apriori(transactions, min_support=0.02):
    """Run Apriori algorithm"""
    # 1-itemsets
    C1 = []
    for t in transactions:
        for item in t:
            if [item] not in C1:
                C1.append([item])
    C1 = [set(c) for c in C1]

    L1, support_data = get_frequent_itemsets(transactions, C1, min_support)
    L = [L1]

    k = 2
    while len(L[k-2]) > 0:
        Ck = generate_candidates(L[k-2], k)
        Lk, supK = get_frequent_itemsets(transactions, Ck, min_support)
        support_data.update(supK)
        if len(Lk) == 0:
            break
        L.append(Lk)
        k += 1

    return L, support_data

def generate_apriori_rules(L, support_data, min_conf=0.5):
    """Generate association rules from Apriori"""
    rules = []
    for i in range(1, len(L)):
        for freq_set in L[i]:
            for conseq in combinations(freq_set, 1):
                conseq = set(conseq)
                antecedent = freq_set - conseq
                if len(antecedent) == 0:
                    continue
                conf = support_data[frozenset(freq_set)] / support_data[frozenset(antecedent)]
                lift = conf / support_data[frozenset(conseq)]
                if conf >= min_conf:
                    rules.append({
                        "antecedent": list(antecedent),
                        "consequent": list(conseq),
                        "support": support_data[frozenset(freq_set)],
                        "confidence": conf,
                        "lift": lift,
                        "algorithm": "Apriori"
                    })
    
    rules.sort(key=lambda x: x['confidence'], reverse=True)
    return rules

# ========================
# TF-IDF FUNCTIONS
# ========================

def generate_tfidf_rules(df, min_support=0.02, min_confidence=0.5):
    """Generate rules using TF-IDF approach"""
    # Replace spaces in item names with underscores for TF-IDF
    df_processed = df.applymap(lambda x: x.replace(" ", "_") if isinstance(x, str) and x != "nan" else x)
    
    # Create documents (each transaction as string of items)
    transactions = df_processed.astype(str).apply(lambda x: ' '.join([item for item in x if item != "nan"]), axis=1)
    
    # TF-IDF transformation
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(transactions)
    items = vectorizer.get_feature_names_out()
    
    # Convert to DataFrame
    tfidf_df = pd.DataFrame(X.toarray(), columns=items)
    binary_df = (tfidf_df > 0).astype(int)
    n_transactions = len(binary_df)
    
    def compute_rule(item_a, item_b):
        support_a = binary_df[item_a].sum() / n_transactions
        support_b = binary_df[item_b].sum() / n_transactions
        support_ab = ((binary_df[item_a] & binary_df[item_b]).sum()) / n_transactions
        
        if support_a == 0 or support_b == 0 or support_ab < min_support:
            return None
            
        confidence = support_ab / support_a
        if confidence < min_confidence:
            return None
            
        lift = confidence / support_b
        
        return {
            "antecedent": [item_a.replace("_", " ")],
            "consequent": [item_b.replace("_", " ")],
            "support": round(support_ab, 4),
            "confidence": round(confidence, 4),
            "lift": round(lift, 4),
            "algorithm": "TF-IDF"
        }
    
    # Generate rules
    rules = []
    for i, item_a in enumerate(items):
        for j, item_b in enumerate(items):
            if i != j:
                rule = compute_rule(item_a, item_b)
                if rule:
                    rules.append(rule)
    
    # Sort by confidence descending
    rules.sort(key=lambda x: x['confidence'], reverse=True)
    return rules, items

# ========================
# RECOMMENDATION FUNCTIONS
# ========================

def get_recommendations_for_cart(cart_items, rules, top_n=8):
    """Get recommendations based on cart items"""
    recommendations = []
    cart_items_set = set([item.lower().strip() for item in cart_items])
    
    for rule in rules:
        # Convert antecedent to lowercase for comparison
        antecedent_set = set([item.lower().strip() for item in rule['antecedent']])
        consequent_items = [item for item in rule['consequent']]
        
        # Check if any cart item matches the antecedent
        if antecedent_set.intersection(cart_items_set):
            # Check if consequent is not already in cart
            consequent_lower = set([item.lower().strip() for item in consequent_items])
            if not consequent_lower.intersection(cart_items_set):
                recommendations.append({
                    'item': ', '.join(consequent_items),
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'support': rule['support'],
                    'based_on': ', '.join(rule['antecedent']),
                    'algorithm': rule['algorithm']
                })
    
    # Remove duplicates and sort by confidence
    seen = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec['item'] not in seen:
            seen.add(rec['item'])
            unique_recommendations.append(rec)
    
    return sorted(unique_recommendations, key=lambda x: x['confidence'], reverse=True)[:top_n]

# ========================
# STREAMLIT APP
# ========================

def main():
    st.markdown('<h1 class="main-header">🛒 Smart Shopping Cart</h1>', unsafe_allow_html=True)
    st.markdown("**Add items to your cart and get intelligent product recommendations powered by machine learning!**")
    
    # Initialize session state for cart
    if 'cart' not in st.session_state:
        st.session_state.cart = []
    if 'rules' not in st.session_state:
        st.session_state.rules = []
    if 'all_items' not in st.session_state:
        st.session_state.all_items = []
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Setup")
        
        # Algorithm selection
        algorithm = st.selectbox(
            "🤖 Choose Recommendation Algorithm",
            ["Apriori", "TF-IDF"],
            help="Select the algorithm for generating product recommendations"
        )
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload Transaction Data",
            type=['csv', 'xlsx'],
            help="Upload a CSV or Excel file with historical transaction data"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Data loaded: {len(df)} transactions")
                
                # Show data preview
                with st.expander("📋 Preview Data"):
                    st.dataframe(df.head(3), use_container_width=True)
                
                # Parameters
                st.subheader("⚙️ Algorithm Settings")
                min_support = st.slider(
                    "Minimum Support", 
                    min_value=0.005, 
                    max_value=0.3, 
                    value=0.02, 
                    step=0.005,
                    help="Items must appear together in at least this % of transactions"
                )
                
                min_confidence = st.slider(
                    "Minimum Confidence", 
                    min_value=0.1, 
                    max_value=0.9, 
                    value=0.4, 
                    step=0.05,
                    help="Recommendation strength threshold"
                )
                
                # Extract all unique items for the item selector
                all_items = set()
                for col in df.columns:
                    items_in_col = df[col].dropna().astype(str)
                    for item in items_in_col:
                        if item != "nan" and item.strip() != "" and item.lower() != "none":
                            all_items.add(item.strip())
                st.session_state.all_items = sorted(list(all_items))
                
                analyze_button = st.button(
                    f"🚀 Initialize {algorithm} Recommendations", 
                    type="primary", 
                    use_container_width=True
                )
                
                if analyze_button:
                    with st.spinner(f"🔄 Training {algorithm} model..."):
                        try:
                            if algorithm == "Apriori":
                                transactions = load_transactions_from_df(df)
                                if len(transactions) == 0:
                                    st.error("❌ No valid transactions found!")
                                    return
                                L, support_data = apriori(transactions, min_support)
                                rules = generate_apriori_rules(L, support_data, min_confidence)
                            else:
                                rules, items = generate_tfidf_rules(df, min_support, min_confidence)
                            
                            if len(rules) == 0:
                                st.warning(f"⚠️ No rules found. Try lowering the thresholds.")
                                return
                            
                            st.session_state.rules = rules
                            st.session_state.analysis_complete = True
                            st.success(f"✅ {algorithm} model ready! Found {len(rules)} recommendation patterns.")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            return
                
                # Model status
                if st.session_state.analysis_complete and st.session_state.rules:
                    st.markdown("---")
                    st.markdown("### 📊 Model Status")
                    st.success("🟢 Model Ready")
                    st.metric("Total Rules", len(st.session_state.rules))
                    st.metric("Algorithm", st.session_state.rules[0]['algorithm'] if st.session_state.rules else "None")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
    
    # Main content
    if not uploaded_file:
        st.info("👈 Please upload transaction data and initialize the recommendation system")
        
        # Sample data format
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 Expected Data Format")
            sample_data = pd.DataFrame({
                'Item1': ['milk', 'bread', 'eggs', 'butter'],
                'Item2': ['bread', 'milk', 'milk', 'milk'],
                'Item3': ['butter', 'eggs', 'bread', 'bread'],
                'Item4': ['', '', 'butter', '']
            })
            st.dataframe(sample_data, use_container_width=True)
            st.caption("Each row = transaction, each column = item position")
        
        with col2:
            st.subheader("🎯 How It Works")
            st.markdown("""
            1. **Upload** your transaction data
            2. **Choose** Apriori or TF-IDF algorithm  
            3. **Initialize** the recommendation system
            4. **Add items** to your cart
            5. **Get** intelligent recommendations!
            """)
        
        return
    
    if not st.session_state.analysis_complete:
        st.info("👈 Please initialize the recommendation system first")
        return
    
    # Shopping Cart Interface
    st.header("🛍️ Shopping Cart & Recommendations")
    
    # Split into two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🛒 Your Cart")
        
        # Item selector
        if st.session_state.all_items:
            selected_item = st.selectbox(
                "Add item to cart:",
                ["Select an item..."] + st.session_state.all_items,
                key="item_selector"
            )
            
            col_add, col_clear = st.columns(2)
            
            with col_add:
                if st.button("➕ Add Item", use_container_width=True) and selected_item != "Select an item...":
                    if selected_item not in st.session_state.cart:
                        st.session_state.cart.append(selected_item)
                        st.success(f"Added '{selected_item}'!")
                        st.rerun()
                    else:
                        st.warning("Already in cart!")
            
            with col_clear:
                if st.button("🗑️ Clear All", use_container_width=True):
                    st.session_state.cart = []
                    st.success("Cart cleared!")
                    st.rerun()
        
        # Display current cart
        if st.session_state.cart:
            st.markdown(f"**{len(st.session_state.cart)} items in cart:**")
            for i, item in enumerate(st.session_state.cart):
                col_item, col_remove = st.columns([4, 1])
                with col_item:
                    st.markdown(f'<div class="cart-item"><strong>{item}</strong></div>', unsafe_allow_html=True)
                with col_remove:
                    if st.button("❌", key=f"remove_{i}", help=f"Remove {item}"):
                        st.session_state.cart.remove(item)
                        st.rerun()
        else:
            st.markdown('<div class="empty-cart">🛒<br>Your cart is empty<br><small>Add items to see recommendations</small></div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("🎯 Recommended for You")
        
        if st.session_state.cart and st.session_state.rules:
            recommendations = get_recommendations_for_cart(
                st.session_state.cart, 
                st.session_state.rules, 
                top_n=8
            )
            
            if recommendations:
                st.markdown(f"**Based on items in your cart, we recommend:**")
                
                for i, rec in enumerate(recommendations, 1):
                    confidence_percent = int(rec['confidence'] * 100)
                    
                    # Add to cart button for recommended item
                    rec_col1, rec_col2 = st.columns([3, 1])
                    
                    with rec_col1:
                        st.markdown(f"""
                        <div class="recommendation">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <strong style="font-size: 1.1em;">#{i}. {rec['item']}</strong>
                                <span class="algorithm-badge">{rec['algorithm']}</span>
                            </div>
                            <div style="font-size: 0.9em; color: #666; margin-bottom: 0.5rem;">
                                Because you have: <strong>{rec['based_on']}</strong>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {confidence_percent}%"></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85em; color: #555;">
                                <span><strong>Confidence:</strong> {confidence_percent}%</span>
                                <span><strong>Lift:</strong> {rec['lift']:.2f}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with rec_col2:
                        if rec['item'] not in st.session_state.cart:
                            if st.button(f"➕", key=f"add_rec_{i}", help=f"Add {rec['item']} to cart"):
                                st.session_state.cart.append(rec['item'])
                                st.success(f"Added {rec['item']}!")
                                st.rerun()
                        else:
                            st.markdown("✅")
            else:
                st.info("🤔 No recommendations available for current cart items. Try adding different products!")
        
        elif st.session_state.cart:
            st.info("🔄 Analyzing cart items for recommendations...")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #666;">
                <h3>👋 Welcome!</h3>
                <p>Add some items to your cart to see personalized recommendations</p>
                <p><small>Our AI will suggest products that other customers frequently buy together</small></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Cart summary
    if st.session_state.cart:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h3>🛒 Cart Items</h3>
                <h2>{len(st.session_state.cart)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            recs = get_recommendations_for_cart(st.session_state.cart, st.session_state.rules, top_n=50)
            avg_conf = np.mean([r['confidence'] for r in recs]) if recs else 0
            st.markdown(f"""
            <div class="metric-container">
                <h3>🎯 Avg Confidence</h3>
                <h2>{avg_conf:.0%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <h3>💡 Available Recs</h3>
                <h2>{len(recs) if 'recs' in locals() else 0}</h2>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()