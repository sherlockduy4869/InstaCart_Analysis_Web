import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import os


fig, ax = plt.subplots(figsize=(14, 6))

st.write("""
# InstaCard Products Bundle Prediction Application

This app predicts the **Products Bundle** for InstaCart platform !
""")

#DATA SOURCE
DATA_SOURCE = "https://drive.google.com/uc?id="

st.sidebar.header('User Input Parameters')

aisles = pd.read_csv(DATA_SOURCE + "1BnskfcHBFUTTvj0d1FMoErTw99eOx-bE")

list_aisles = aisles['aisle'].to_list()

def user_input_features():
    aisles = st.sidebar.multiselect('Aisles',list_aisles, list_aisles[1:4])
    max_len = st.sidebar.slider('Max length', 2, 5, 3)
    frequency = st.sidebar.slider('Frequency', 100, 1000, 150)
    data = {
        'frequency' : [frequency],
        'max_len': [max_len],
        'aisles' : [aisles]
        }
    features = pd.DataFrame(data)
    return features

df_user_input = user_input_features()

submit_button = st.sidebar.button("Run", type="primary")

#pip list --format=freeze | grep -v file:// > requirements.txt

if submit_button:

    url_data = DATA_SOURCE + '1u2LR1VoFCy_DvG1NSQamRQsR7LbGLNuM'

    data_file = 'processed_data.csv'

    gdown.download(url_data, data_file, quiet=False)

    file_path_data = os.getcwd() + '/processed_data.csv'

    data = pd.read_csv(file_path_data)

    #INPUT DATA
    max_len = df_user_input['max_len'][0]
    list_aisle = df_user_input['aisles'][0]
    product_frequency = df_user_input['frequency'][0]

    # DATA PROCESSING

    st.subheader('Up to this four')

    if len(list_aisle) > 0:
        data = data[data['aisle'].isin(list_aisle)]
    else:
        data = data

    st.subheader('Up to this five')

    data_explore = data[['order_id', 'product_name']]

    #Calculating support point
    total_order = data_explore["order_id"].nunique()

    support_point = product_frequency/total_order

    basket = data_explore.groupby('order_id')['product_name'].apply(list).tolist()

    te = TransactionEncoder()
    te_ary = te.fit(basket).transform(basket, sparse=True)

    df = pd.DataFrame.sparse.from_spmatrix(
        te_ary,
        columns=te.columns_
    )

    st.subheader('Up to this 6')

    frequent_itemsets = fpgrowth(df, min_support = support_point, use_colnames=True, max_len = max_len)

    rules = association_rules(
        frequent_itemsets,
        metric = "lift",
        min_threshold = 1.5
    )

    rules = rules.sort_values(by = ["lift", "support"], ascending=False)

    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]

    rules = rules.drop_duplicates(subset=["support"])

    rules["antecedents"] = rules["antecedents"].apply(
        lambda x: list(x) if isinstance(x, (set, frozenset)) else x
    )

    rules["consequents"] = rules["consequents"].apply(
        lambda x: list(x) if isinstance(x, (set, frozenset)) else x
    )

    rules = rules.sort_values(by = ["lift", "support"], ascending=False)

    st.subheader('Products Bundle Result')
    st.write(rules)