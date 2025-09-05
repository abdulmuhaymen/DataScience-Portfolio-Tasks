import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Market Basket Analysis",
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
    }
    .rule-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ========================
# APRIORI ALGORITHM FUNCTIONS
# (Replicated from your model.pynb)
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
        if transaction:  # Only add non-empty transactions
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
            if l1[:k-2] == l2[:k-2]:  # join step
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

def generate_rules(L, support_data, min_conf=0.5):
    """Generate association rules"""
    rules = []
    for i in range(1, len(L)):   # from 2-itemsets onwards
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
                        "antecedent": antecedent,
                        "consequent": conseq,
                        "support": support_data[frozenset(freq_set)],
                        "confidence": conf,
                        "lift": lift
                    })
    
    # Sort by confidence descending
    rules.sort(key=lambda x: x['confidence'], reverse=True)
    return rules

def print_rules_natural(rules, top_n=10):
    """Print rules in natural language format"""
    rule_texts = []
    for r in rules[:top_n]:
        antecedent = ', '.join(sorted(list(r['antecedent'])))
        consequent = ', '.join(sorted(list(r['consequent'])))
        rule_text = (f"If someone buys {antecedent}, "
                    f"then with probability = {r['confidence']:.2f}, "
                    f"they will also buy {consequent}. "
                    f"(Support: {r['support']:.2f}, Lift: {r['lift']:.2f})")
        rule_texts.append(rule_text)
    return rule_texts

# ========================
# STREAMLIT APP
# ========================

def main():
    st.markdown('<h1 class="main-header">🛒 Market Basket Analysis</h1>', unsafe_allow_html=True)
    st.markdown("**Upload your transaction data and discover hidden patterns in customer purchasing behavior!**")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Transaction Data",
            type=['csv', 'xlsx'],
            help="Upload a CSV or Excel file with transaction data"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")
                
                # Show data preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(), use_container_width=True)
                
                # Parameters
                st.subheader("Algorithm Parameters")
                min_support = st.slider(
                    "Minimum Support", 
                    min_value=0.01, 
                    max_value=0.5, 
                    value=0.02, 
                    step=0.01,
                    help="Minimum support threshold for frequent itemsets"
                )
                
                min_confidence = st.slider(
                    "Minimum Confidence", 
                    min_value=0.1, 
                    max_value=1.0, 
                    value=0.45, 
                    step=0.05,
                    help="Minimum confidence threshold for association rules"
                )
                
                max_rules = st.number_input(
                    "Maximum Rules to Display", 
                    min_value=5, 
                    max_value=100, 
                    value=20
                )
                
                analyze_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
        else:
            analyze_button = False
    
    # Main content
    if uploaded_file is None:
        st.info("👈 Please upload a transaction data file to begin analysis")
        
        # Sample data format
        st.subheader("📝 Expected Data Format")
        st.write("Your data should have transactions in rows and items in columns:")
        
        sample_data = pd.DataFrame({
            'Item1': ['milk', 'bread', 'eggs', 'butter', 'yogurt'],
            'Item2': ['bread', 'milk', 'milk', 'milk', 'milk'],
            'Item3': ['butter', 'eggs', 'bread', 'bread', 'bread'],
            'Item4': ['', '', 'butter', '', ''],
            'Item5': ['', '', '', '', '']
        })
        st.dataframe(sample_data, use_container_width=True)
        st.caption("Each row = one transaction, each column = one item position")
        
    elif analyze_button:
        with st.spinner("🔄 Processing transactions..."):
            try:
                # Load and process data
                transactions = load_transactions_from_df(df)
                
                if len(transactions) == 0:
                    st.error("❌ No valid transactions found in the data!")
                    return
                
                st.success(f"✅ Loaded {len(transactions)} valid transactions")
                
                # Run Apriori algorithm
                with st.spinner("⚙️ Running Apriori algorithm..."):
                    L, support_data = apriori(transactions, min_support)
                    rules = generate_rules(L, support_data, min_confidence)
                
                if len(rules) == 0:
                    st.warning(f"⚠️ No association rules found with minimum confidence {min_confidence}. Try lowering the confidence threshold.")
                    return
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>📊 Transactions</h3>
                        <h2>{len(transactions):,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    total_items = len(set().union(*transactions)) if transactions else 0
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>📦 Unique Items</h3>
                        <h2>{total_items:,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>📈 Rules Found</h3>
                        <h2>{len(rules):,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    avg_confidence = np.mean([r['confidence'] for r in rules]) if rules else 0
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>🎯 Avg Confidence</h3>
                        <h2>{avg_confidence:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Create tabs for results
                tab1, tab2, tab3 = st.tabs(["🎯 Association Rules", "📊 Visualizations", "💾 Export Data"])
                
                with tab1:
                    st.subheader("🎯 Top Association Rules (Natural Language)")
                    
                    # Display rules in your exact format
                    rules_to_show = rules[:max_rules]
                    rule_texts = print_rules_natural(rules_to_show, len(rules_to_show))
                    
                    for i, rule_text in enumerate(rule_texts, 1):
                        st.markdown(f"""
                        <div class="rule-card">
                            <strong>Rule #{i}:</strong><br>
                            {rule_text}
                        </div>
                        """, unsafe_allow_html=True)
                
                with tab2:
                    st.subheader("📊 Rules Visualization")
                    
                    if rules:
                        # Prepare data for visualization
                        viz_data = []
                        for i, rule in enumerate(rules[:20]):  # Top 20 for visualization
                            antecedent = ', '.join(sorted(list(rule['antecedent'])))
                            consequent = ', '.join(sorted(list(rule['consequent'])))
                            viz_data.append({
                                'Rule': f"{antecedent} → {consequent}",
                                'Support': rule['support'],
                                'Confidence': rule['confidence'],
                                'Lift': rule['lift'],
                                'Rule_Number': i + 1
                            })
                        
                        viz_df = pd.DataFrame(viz_data)
                        
                        # Support vs Confidence scatter plot
                        fig1 = px.scatter(
                            viz_df, 
                            x='Support', 
                            y='Confidence',
                            size='Lift',
                            hover_data=['Rule'],
                            title="Support vs Confidence (Bubble size = Lift)",
                            color='Lift',
                            color_continuous_scale='viridis',
                            height=500
                        )
                        fig1.update_layout(showlegend=True)
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Top rules by confidence
                        fig2 = px.bar(
                            viz_df.head(10), 
                            x='Confidence', 
                            y='Rule',
                            title="Top 10 Rules by Confidence",
                            orientation='h',
                            color='Confidence',
                            color_continuous_scale='blues',
                            height=500
                        )
                        fig2.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig2, use_container_width=True)
                
                with tab3:
                    st.subheader("💾 Export Results")
                    
                    # Create downloadable dataframe
                    export_data = []
                    for rule in rules:
                        antecedent = ', '.join(sorted(list(rule['antecedent'])))
                        consequent = ', '.join(sorted(list(rule['consequent'])))
                        export_data.append({
                            'Antecedent': antecedent,
                            'Consequent': consequent,
                            'Support': round(rule['support'], 4),
                            'Confidence': round(rule['confidence'], 4),
                            'Lift': round(rule['lift'], 4),
                            'Natural_Language_Rule': f"If someone buys {antecedent}, then with probability = {rule['confidence']:.2f}, they will also buy {consequent}. (Support: {rule['support']:.2f}, Lift: {rule['lift']:.2f})"
                        })
                    
                    export_df = pd.DataFrame(export_data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="📄 Download CSV",
                            data=csv,
                            file_name=f"association_rules_{min_support}_{min_confidence}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Save rules as pickle (optional)
                        if st.button("💾 Save Analysis", use_container_width=True):
                            analysis_data = {
                                'rules': rules,
                                'frequent_itemsets': L,
                                'support_data': support_data,
                                'parameters': {
                                    'min_support': min_support,
                                    'min_confidence': min_confidence
                                }
                            }
                            st.success("✅ Analysis data prepared for download!")
                    
                    # Show preview of export data
                    st.subheader("📋 Export Preview")
                    st.dataframe(export_df.head(), use_container_width=True)
                
                # Business insights
                st.subheader("💡 Key Business Insights")
                
                if rules:
                    top_rule = rules[0]
                    high_lift_rules = [r for r in rules if r['lift'] > 2.0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🏆 Best Performing Rule")
                        antecedent = ', '.join(sorted(list(top_rule['antecedent'])))
                        consequent = ', '.join(sorted(list(top_rule['consequent'])))
                        st.info(f"**{antecedent}** → **{consequent}**\n\n"
                               f"Confidence: {top_rule['confidence']:.1%}\n"
                               f"Support: {top_rule['support']:.1%}\n"
                               f"Lift: {top_rule['lift']:.2f}")
                    
                    with col2:
                        st.markdown("### 🚀 High Impact Rules")
                        if high_lift_rules:
                            st.success(f"Found {len(high_lift_rules)} rules with lift > 2.0")
                            for rule in high_lift_rules[:3]:
                                ant = ', '.join(sorted(list(rule['antecedent'])))
                                cons = ', '.join(sorted(list(rule['consequent'])))
                                st.write(f"• **{ant}** → **{cons}** (Lift: {rule['lift']:.2f})")
                        else:
                            st.warning("No rules found with lift > 2.0")
                    
                    # Actionable recommendations
                    st.markdown("### 📋 Actionable Recommendations")
                    recommendations = [
                        f"🛒 **Cross-selling**: Place **{', '.join(sorted(list(top_rule['consequent'])))}** near **{', '.join(sorted(list(top_rule['antecedent'])))}**",
                        f"🎯 **Bundle deals**: Create combo offers with frequently associated items",
                        f"📦 **Inventory**: Ensure stock availability of complementary products",
                        f"📧 **Marketing**: Target customers buying **{', '.join(sorted(list(top_rule['antecedent'])))}** with **{', '.join(sorted(list(top_rule['consequent'])))}** promotions"
                    ]
                    
                    for rec in recommendations:
                        st.write(rec)
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.write("Please check your data format and try again.")

if __name__ == "__main__":
    main()