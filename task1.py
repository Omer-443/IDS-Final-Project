import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Smoking Behavior Analysis",
    page_icon="\U0001F6AD",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    data = pd.read_csv('DataSet/smoking.csv')
    data.loc[data['smoke'] == "No", ['amt_weekends', 'amt_weekdays']] = 0
    data.loc[data['smoke'] == "No", 'type'] = "Not Applicable"
    smokers = data[data['smoke'] == "Yes"]
    median_amt_weekends = smokers['amt_weekends'].median()
    median_amt_weekdays = smokers['amt_weekdays'].median()
    mode_type = smokers['type'].mode()[0]
    data.loc[smokers.index, 'amt_weekends'] = smokers['amt_weekends'].fillna(median_amt_weekends)
    data.loc[smokers.index, 'amt_weekdays'] = smokers['amt_weekdays'].fillna(median_amt_weekdays)
    data.loc[smokers.index, 'type'] = smokers['type'].fillna(mode_type)
    return data

data = load_data()
