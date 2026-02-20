import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import gdown
import os

st.write("""
# InstaCard Products Bundle Prediction Application

This app predicts the **Products Bundle** for InstaCart platform !
""")

#GETTING THE LIST OF AISLES

DATA_SOURCE = "https://drive.google.com/uc?id="
aisles = pd.read_csv(DATA_SOURCE + "1BnskfcHBFUTTvj0d1FMoErTw99eOx-bE")
list_aisles = aisles['aisle'].to_list()

st.sidebar.header('User Input Parameters')
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

    #GETTING THE DATA TO PROCESS
    url_data = DATA_SOURCE + '1u2LR1VoFCy_DvG1NSQamRQsR7LbGLNuM'
    data_file = 'processed_data.csv'
    gdown.download(url_data, data_file, quiet=False)
    file_path_data = os.getcwd() + '/processed_data.csv'
    data = pd.read_csv(file_path_data)

    #INPUT DATA
    max_len = df_user_input['max_len'][0]
    list_aisle = df_user_input['aisles'][0]
    product_frequency = df_user_input['frequency'][0]

    # DATA PROCESSING BY SELECTED AISLES
    if len(list_aisle) > 0:
        data = data[data['aisle'].isin(list_aisle)]
    else:
        data = data

    data_explore = data[['order_id', 'product_name']]

    #CALCULATING SUPPORT POINT
    total_order = data_explore["order_id"].nunique()
    support_point = product_frequency/total_order

    #MODEL TRAINING
    basket = data_explore.groupby('order_id')['product_name'].apply(list).tolist()
    te = TransactionEncoder()
    te_ary = te.fit(basket).transform(basket, sparse=True)
    df = pd.DataFrame.sparse.from_spmatrix(
        te_ary,
        columns=te.columns_
    )
    frequent_itemsets = fpgrowth(df, min_support = support_point, use_colnames=True, max_len = max_len)
    rules = association_rules(
        frequent_itemsets,
        metric = "lift",
        min_threshold = 1.5
    )

    #DISPlAYING THE RESULT

    rules = rules.sort_values(by = ["lift", "support"], ascending=False)

    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]

    rules = rules.drop_duplicates(subset=["support"])

    rules["antecedents"] = rules["antecedents"].apply(
        lambda x: list(x) if isinstance(x, (set, frozenset)) else x
    )

    rules["consequents"] = rules["consequents"].apply(
        lambda x: list(x) if isinstance(x, (set, frozenset)) else x
    )

    st.subheader('Products Bundle Result')
    st.write(rules)