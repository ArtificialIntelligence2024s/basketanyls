import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

# Load data
df = pd.read_csv("bread_basket.csv")
df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M')

# Assuming 'period_day' and 'weekday_weekend' columns are created based on 'date_time'
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.weekday

df['month'].replace([i for i in range(1, 12 + 1)], ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], inplace=True)
df['day'].replace([i for i in range(0, 6 + 1)], ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], inplace=True)

st.title('Market Basket Analysis Menggunakan Algortima Apriori')

def get_data(period_day='', weekday_weekend="", month="", day=""):
    data = df.copy()
    filtered = data[
        (data['period_day'] == period_day) &
        (data['weekday_weekend'] == weekday_weekend) &
        (data['month'] == month.title()) &
        (data['day'] == day.title())
    ]
    return filtered if not filtered.empty else "No Result!"


def user_input_features():
    item = st.selectbox('Masukkan Item', df['Item'].unique())
    period_day = st.selectbox('Pilih Waktu', ['Morning', 'Afternoon', 'Evening', 'Night'])
    weekday_weekend = st.selectbox('Pilih Hari', ['Weekday', 'Weekend'])
    month = st.select_slider('Pilih Bulan', ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    day = st.select_slider('Pilih Hari', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    return item, period_day, weekday_weekend, month, day

item, period_day, weekday_weekend, month, day = user_input_features()

data = get_data(period_day.lower(), weekday_weekend.lower(), month, day)

def encode(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

if isinstance(data, pd.DataFrame):
    item_count = data.groupby(['Transaction', 'Item']).size().reset_index(name='count')
    item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)

    support = 0.01
    frequent_itemsets = apriori(item_count_pivot, min_support=support, use_colnames=True)

    metric = "lift"
    min_threshold = 1

    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules.sort_values(by='confidence', ascending=False, inplace=True)

def parse_list(x):
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)
    
def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()
    data["antecedents"] = data["antecedents"].apply(lambda x: list(x))
    data["consequents"] = data["consequents"].apply(lambda x: list(x))
    if not data.empty:
        filtered_data = data.loc[data["antecedents"].apply(lambda x: x == item_antecedents)]
        print("Filtered Data:")
        print(filtered_data)
        if not filtered_data.empty and len(filtered_data) > 0:
            print("First row of filtered data:")
            print(filtered_data.iloc[0, :])
            return list(filtered_data.iloc[0, :])
    return []
if type(data) != type("No Result!"):
    st.markdown("Hasil Rekomendasi")
    recommended_items = return_item_df(item)
    if len(recommended_items) >= 2:
        st.success(f"Item: **{item}**, maka membeli **{recommended_items[1]}** secara bersamaan")
    else:
        st.warning("Tidak ada rekomendasi yang ditemukan.")
else:
    st.warning("Tidak ada hasil yang cocok dengan kriteria yang dipilih.")
