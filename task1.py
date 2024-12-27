import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

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
st.sidebar.title("Navigation")
pages = [
    "Overview",
    
]
selected_page = st.sidebar.radio("Select a Page", pages)
if selected_page == "Overview":
    st.title("\U0001F6AD Smoking Dataset Overview")
    total_records = len(data)
    smokers = len(data[data['smoke'] == "Yes"])
    non_smokers = len(data[data['smoke'] == "No"])
    avg_age = round(data['age'].mean(), 1)

    # Display metrics
    st.metric("Total Records", total_records)
    st.metric("Smokers", smokers)
    st.metric("Non-Smokers", non_smokers)
    st.metric("Average Age", f"{avg_age} years")

    fig = px.pie(
        names=["Smokers", "Non-Smokers"],
        values=[smokers, non_smokers],
        title="Smoking Proportion in Dataset",
        color_discrete_sequence=px.colors.sequential.RdBu,
    )
    st.plotly_chart(fig)
