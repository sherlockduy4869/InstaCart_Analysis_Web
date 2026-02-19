import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import os
from pyspark.sql import SparkSession


spark = SparkSession.builder \
    .appName("instacart_merge") \
    .getOrCreate()

fig, ax = plt.subplots(figsize=(14, 6))

st.write("""
# InstaCard Products Bundle Prediction Application

This app predicts the **Products Bundle** for InstaCart platform !
""")

#DATA SOURCE
DATA_SOURCE = "https://drive.google.com/uc?id="

st.sidebar.header('User Input Parameters')

url_departments = DATA_SOURCE + "1mBix_Zbg67I1foEsljFm7Cf23H_SFhcI"
url_aisles = DATA_SOURCE + "1BnskfcHBFUTTvj0d1FMoErTw99eOx-bE"
url_orders = DATA_SOURCE + "1Mb1BvHirYGVbsKiddfJR6uaRrBQt4KwU"
url_order_products_prior = DATA_SOURCE + "1M8fuFMODgT_6K_XeTzR2Upg3fDPbKN66"
url_order_products_train = DATA_SOURCE + "1Egfno5jVrQXCkrhEgdPrQR2akqs7U5fJ"
url_products = DATA_SOURCE + "18weuttpH8e1NaHDWINx6etGFj92R0_50"

departments_file = "departments.csv"
aisles_file = "aisles.csv"
orders_file = "orders.csv"
order_products_prior_file = "order_products_prior.csv"
order_products_train_file = "order_products_train.csv"
products_file = "products.csv"

gdown.download(url_departments, departments_file, quiet=False)
gdown.download(url_aisles, aisles_file, quiet=False)
gdown.download(url_orders, orders_file, quiet=False)
gdown.download(url_order_products_prior, order_products_prior_file, quiet=False)
gdown.download(url_order_products_train, order_products_train_file, quiet=False)
gdown.download(url_products, products_file, quiet=False)

order_products_prior = spark.read.csv(order_products_train_file, header=True, inferSchema=True)
order_products_train = spark.read.csv(order_products_train_file, header=True, inferSchema=True)
departments = spark.read.csv(departments_file, header=True, inferSchema=True)
aisles = spark.read.csv(aisles_file, header=True, inferSchema=True)
orders = spark.read.csv(orders_file, header=True, inferSchema=True)
products = spark.read.csv(products_file, header=True, inferSchema=True)

departments_row = departments.select("department").collect()
list_departments = [department_row['department'] for department_row in departments_row]

aisles_row = aisles.select("aisle").collect()
list_aisles = [aisle_row['aisle'] for aisle_row in aisles_row]

order_products = order_products_train.union(order_products_prior)
st.subheader('Up to this four')
data = order_products.join(products, on="product_id", how="left") \
                     .join(aisles, on="aisle_id", how="left") \
                     .join(departments, on="department_id", how="left") \
                     .join(orders, on="order_id", how="left")
st.subheader('Up to this five')
# def user_input_features():
#     departments = st.sidebar.multiselect('Derpartments', list_departments, list_departments[1:4])
#     aisles = st.sidebar.multiselect('Aisles',list_aisles, list_aisles[1:4])
#     max_len = st.sidebar.slider('Max length', 2, 5, 3)
#     frequency = st.sidebar.slider('Frequency', 100, 1000, 150)
#     data = {
#         'frequency' : [frequency],
#         'max_len': [max_len],
#         'departments' : [departments],
#         'aisles' : [aisles]
#         }
#     features = pd.DataFrame(data)
#     return features

# df_user_input = user_input_features()

# submit_button = st.sidebar.button("Run", type="primary")

# #pip list --format=freeze | grep -v file:// > requirements.txt

# if submit_button:

    

#     file_path_orders = os.getcwd() + '/orders.csv'
#     file_path_order_products_prior = os.getcwd() + '/order_products_prior.csv'

#     st.subheader('Test file path on cloud')
#     st.write(os.path.exists(file_path_orders))
#     st.write(os.path.exists(file_path_order_products_prior))

#     orders = pd.read_csv(file_path_orders)
#     order_products_prior = pd.read_csv(file_path_order_products_prior)

#     st.subheader('Up to this one')

#     #INPUT DATA
#     max_len = df_user_input['max_len'][0]
#     list_department = df_user_input['departments'][0]
#     list_aisle = df_user_input['aisles'][0]
#     product_frequency = df_user_input['frequency'][0]

#     order_products = pd.concat([order_products_train, order_products_prior])

#     data = order_products.merge(orders, on = "order_id", how = "left")

#     data = data.drop_duplicates(subset=["user_id","product_id"])

#     data = order_products.merge(products, on = "product_id", how = "left")

#     data = data.merge(aisles, on = "aisle_id", how = "left")

#     data = data.merge(departments, on = "department_id", how = "left")

#     sns.countplot(x='order_dow', data=data, color='teal', ax=ax)

#     ax.set_title('Busiest Days of the Week')
#     ax.set_xlabel('Day of Week (0 = Sunday)')
#     ax.set_ylabel('Total Orders')
#     ax.ticklabel_format(style='plain', axis='y')
#     st.pyplot(fig)

#     # DATA PROCESSING

#     st.subheader('Up to this four')

#     if len(list_department) > 0:
#         data = data[data['department'].isin(list_department)]
#     elif len(list_aisle) > 0:
#         data = data[data['aisle'].isin(list_aisle)]
#     else:
#         data = data

#     data_explore = data[['order_id', 'product_name']]

#     #Calculating support point
#     total_order = data_explore["order_id"].nunique()

#     support_point = product_frequency/total_order

#     basket = data_explore.groupby('order_id')['product_name'].apply(list).tolist()

#     te = TransactionEncoder()
#     te_ary = te.fit(basket).transform(basket)
#     df = pd.DataFrame(te_ary, columns=te.columns_)

#     frequent_itemsets = fpgrowth(df, min_support = support_point, use_colnames=True, max_len = max_len)

#     rules = association_rules(
#         frequent_itemsets,
#         metric = "lift",
#         min_threshold = 1.5
#     )

#     rules = rules.sort_values(by = ["lift", "support"], ascending=False)

#     rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]

#     rules = rules.drop_duplicates(subset=["support"])

#     rules["antecedents"] = rules["antecedents"].apply(
#         lambda x: list(x) if isinstance(x, (set, frozenset)) else x
#     )

#     rules["consequents"] = rules["consequents"].apply(
#         lambda x: list(x) if isinstance(x, (set, frozenset)) else x
#     )

#     rules = rules.sort_values(by = ["lift", "support"], ascending=False)

#     st.subheader('Products Bundle Result')
#     st.write(rules)