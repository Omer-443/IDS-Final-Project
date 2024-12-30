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
    "Smoking by Gender",
    "Age Distribution",
    "Income Analysis",
    "Smoking Type Distribution",
    "Weekday vs Weekend Smoking",
    "Regional Smoking Patterns",
    "Correlation Analysis",
]
selected_page = st.sidebar.radio("Select a Page", pages)
if selected_page == "Overview":
    st.title("\U0001F6AD Smoking Dataset Overview")
    total_records = len(data)
    smokers = len(data[data['smoke'] == "Yes"])
    non_smokers = len(data[data['smoke'] == "No"])
    avg_age = round(data['age'].mean(), 1)

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

elif selected_page == "Smoking by Gender":
    st.title("Smoking Behavior by Gender")
    gender_smoking = data.groupby(['gender', 'smoke']).size().reset_index(name='count')
    fig = px.bar(
        gender_smoking,
        x="gender",
        y="count",
        color="smoke",
        title="Smoking by Gender",
        barmode="group",
        color_discrete_map={"Yes": "red", "No": "green"},
    )
    st.plotly_chart(fig)

elif selected_page == "Age Distribution":
    st.title("Age Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data, x="age", hue="smoke", kde=True, bins=20, ax=ax)
    ax.set_title("Age Distribution of Smokers and Non-Smokers")
    st.pyplot(fig)

elif selected_page == "Income Analysis":
    st.title("Income vs Smoking Behavior")
    income_smoking = data.groupby(['gross_income', 'smoke']).size().reset_index(name='count')
    fig = px.bar(
        income_smoking,
        x="gross_income",
        y="count",
        color="smoke",
        title="Smoking by Income Bracket",
        barmode="group",
        color_discrete_map={"Yes": "red", "No": "green"},
    )
    st.plotly_chart(fig)

elif selected_page == "Smoking Type Distribution":
    st.title("Smoking Type Distribution")
    fig = px.pie(
        data,
        names="type",
        title="Distribution of Smoking Types",
        color_discrete_sequence=px.colors.sequential.RdBu,
    )
    st.plotly_chart(fig)

elif selected_page == "Weekday vs Weekend Smoking":
    st.title("Weekday vs Weekend Smoking")
    fig = px.scatter(
        data,
        x="amt_weekdays",
        y="amt_weekends",
        color="smoke",
        title="Weekday vs Weekend Smoking",
        labels={"amt_weekdays": "Weekday Smoking", "amt_weekends": "Weekend Smoking"},
    )
    st.plotly_chart(fig)

elif selected_page == "Regional Smoking Patterns":
    st.title("Smoking Patterns by Region")
    region_smoking = data.groupby("region")['smoke'].value_counts(normalize=True).unstack()
    fig = px.bar(
        region_smoking,
        barmode="stack",
        title="Smoking Patterns Across Regions",
        labels={"value": "Proportion"},
    )
    st.plotly_chart(fig)

elif selected_page == "Correlation Analysis":
    st.title("Correlation Between Features")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Include only numeric columns
    correlation_matrix = numeric_data.corr()  # Calculate correlation
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)    