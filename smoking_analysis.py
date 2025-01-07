import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# App Configuration
st.set_page_config(
    page_title="Smoking Behavior Analysis",
    page_icon=":no_smoking:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv('DataSet/smoking.csv')  # Load the provided dataset
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

# Sidebar - Filters
st.sidebar.title("Filters")
gender_filter = st.sidebar.multiselect("Filter by Gender", options=data['gender'].unique(), default=data['gender'].unique())
age_filter = st.sidebar.slider("Filter by Age", int(data['age'].min()), int(data['age'].max()), (20, 50))
region_filter = st.sidebar.multiselect("Filter by Region", options=data['region'].unique(), default=data['region'].unique())

filtered_data = data[
    (data['gender'].isin(gender_filter)) & 
    (data['age'].between(age_filter[0], age_filter[1])) & 
    (data['region'].isin(region_filter))
]

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = [
    "Overview & Demographics", 
    "Smoking Behavior", 
    "Trends & Correlations", 
    "Advanced Analysis", 
    "Income and Smoking Type", 
    "Prediction"
]
selected_page = st.sidebar.radio("Select a Page", pages)

# PAGE 1: Overview & Demographics
if selected_page == "Overview & Demographics":
    st.title("🚭 Smoking Dataset: Overview & Demographics")
    
    # Metrics
    total_records = len(filtered_data)
    smokers = len(filtered_data[filtered_data['smoke'] == "Yes"])
    non_smokers = len(filtered_data[filtered_data['smoke'] == "No"])
    avg_age = round(filtered_data['age'].mean(), 1)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", total_records)
    col2.metric("Smokers", smokers, delta=f"{(smokers / total_records) * 100:.1f}% of Total")
    col3.metric("Non-Smokers", non_smokers, delta=f"{(non_smokers / total_records) * 100:.1f}% of Total")
    col4.metric("Average Age", f"{avg_age} years")
    
    # Gender Distribution
    gender_data = (
        filtered_data.groupby(['gender', 'smoke'])
        .size()
        .reset_index(name='count')
    )
    gender_totals = gender_data.groupby('gender')['count'].sum().reset_index(name='total')
    gender_data = gender_data.merge(gender_totals, on='gender')
    gender_data['percentage'] = (gender_data['count'] / gender_data['total']) * 100

    fig = px.bar(
        gender_data,
        x="gender",
        y="count",
        color="smoke",
        text=gender_data["percentage"].apply(lambda x: f"{x:.1f}%"),
        title="Smoking Behavior by Gender",
        labels={"count": "Number of Individuals", "gender": "Gender", "smoke": "Smoking Status"},
        barmode="group",
        color_discrete_map={"Yes": "#FF6F61", "No": "#6BAED6"},
        hover_data={"percentage": True, "total": True},
    )
    fig.update_traces(textposition="outside", marker=dict(line=dict(width=0.5, color="black")))
    fig.update_layout(
        title=dict(font_size=20, x=0.5),
        xaxis=dict(title="Gender", tickangle=-45),
        yaxis=dict(title="Number of Individuals"),
        legend=dict(title="Smoking Status", orientation="h", x=0.5, xanchor="center"),
        plot_bgcolor="#F9F9F9",
    )
    st.plotly_chart(fig)

# PAGE 2: Smoking Behavior
elif selected_page == "Smoking Behavior":
    st.title("🚬 Smoking Behavior Analysis")
    tabs = st.tabs(["By Income", "By Education", "Smoking Types"])
    
    with tabs[0]:
        st.subheader("Income vs Smoking Behavior")
        income_smoking = filtered_data.groupby(['gross_income', 'smoke']).size().reset_index(name='count')
        fig = px.bar(
            income_smoking,
            x="gross_income",
            y="count",
            color="smoke",
            title="Smoking by Income Bracket",
            barmode="group",
            color_discrete_map={"Yes": "red", "No": "green"}
        )
        st.plotly_chart(fig)

    with tabs[1]:
        st.subheader("Smoking by Education Level")
        education_smoking = filtered_data.groupby(['highest_qualification', 'smoke']).size().reset_index(name='count')
        fig = px.bar(
            education_smoking,
            x="highest_qualification",
            y="count",
            color="smoke",
            title="Smoking by Education Level",
            barmode="group",
        )
        st.plotly_chart(fig)

    with tabs[2]:
        st.subheader("Smoking Type Distribution")
        fig = px.pie(
            filtered_data,
            names="type",
            title="Distribution of Smoking Types",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig)

# PAGE 3: Trends & Correlations
elif selected_page == "Trends & Correlations":
    st.title("📈 Trends & Correlations")
    tabs = st.tabs(["Age Distribution", "Smoking Frequency", "Correlation Analysis"])
    
    with tabs[0]:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_data, x="age", hue="smoke", kde=True, bins=20, ax=ax)
        ax.set_title("Age Distribution of Smokers and Non-Smokers")
        st.pyplot(fig)
    
    with tabs[1]:
        st.subheader("Smoking Frequency: Weekdays vs Weekends")
        fig = px.scatter(
            filtered_data,
            x="amt_weekdays",
            y="amt_weekends",
            color="smoke",
            title="Smoking Frequency: Weekdays vs Weekends",
            color_discrete_map={"Yes": "red", "No": "green"}
        )
        st.plotly_chart(fig)

    with tabs[2]:
        st.subheader("Correlation Between Features")
        numeric_data = filtered_data.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# PAGE 4: Advanced Analysis
elif selected_page == "Advanced Analysis":
    st.title("🧠 Advanced Smoking Analysis")
    tabs = st.tabs(["Age Group Trends", "Regional Patterns", "Marital Status & Smoking"])
    
    with tabs[0]:
        st.subheader("Smoking Trends by Age Groups")
        # Check if 'age' column has any missing data
        if 'age' in filtered_data.columns and not filtered_data['age'].isnull().any():
            # Categorize the 'age' column into age groups
            filtered_data['age_group'] = pd.cut(filtered_data['age'], bins=[0, 18, 30, 50, 70, 100], labels=["<18", "18-30", "30-50", "50-70", "70+"])
            age_group_smoking = filtered_data.groupby(['age_group', 'smoke']).size().reset_index(name='count')
            
            # Check if the data exists before plotting
            if not age_group_smoking.empty:
                fig = px.bar(
                    age_group_smoking,
                    x="age_group",
                    y="count",
                    color="smoke",
                    title="Smoking Trends by Age Groups",
                    barmode="group",
                    color_discrete_map={"Yes": "red", "No": "green"}
                )
                st.plotly_chart(fig)
            else:
                st.write("No data available for age group analysis.")
        else:
            st.write("Missing or invalid data in the 'age' column.")


    with tabs[1]:
        st.subheader("Smoking Patterns by Region")
        region_smoking = filtered_data.groupby("region")['smoke'].value_counts().unstack().reset_index()
        region_smoking.fillna(0, inplace=True)
        fig = px.bar(
            region_smoking,
            x="region",
            y=["Yes", "No"],
            title="Smoking Patterns by Region",
            labels={"region": "Region", "value": "Count", "smoke": "Smoking Status"},
            color_discrete_map={"Yes": "red", "No": "green"}
        )
        st.plotly_chart(fig)

    with tabs[2]:
        st.subheader("Smoking Behavior by Marital Status")
        marital_status_smoking = filtered_data.groupby("marital_status")['smoke'].value_counts().unstack().reset_index()
        marital_status_smoking.fillna(0, inplace=True)
        fig = px.bar(
            marital_status_smoking,
            x="marital_status",
            y=["Yes", "No"],
            title="Smoking Behavior by Marital Status",
            labels={"marital_status": "Marital Status", "value": "Count", "smoke": "Smoking Status"},
            color_discrete_map={"Yes": "red", "No": "green"}
        )
        st.plotly_chart(fig)

# PAGE 5: Income and Smoking Type
elif selected_page == "Income and Smoking Type":
    st.title("💸 Income vs Smoking Type")
    income_type_data = filtered_data.groupby(['gross_income', 'type']).size().reset_index(name='count')
    fig = px.bar(
        income_type_data,
        x="gross_income",
        y="count",
        color="type",
        title="Income and Smoking Type Distribution",
        color_discrete_map={"Cigarettes": "red", "Vaping": "blue", "Other": "green", "Not Applicable": "grey"},
        labels={"gross_income": "Income Bracket", "count": "Count"}
    )
    st.plotly_chart(fig)
# Function to clean income data
def clean_income(income):
    try:
        if 'Under' in income:
            return int(income.split(' ')[1].replace(',', '').strip())
        elif 'Above' in income:
            return int(income.split(' ')[1].replace(',', '').strip())
        elif 'to' in income:
            income_values = income.split(' to ')
            lower_value = int(income_values[0].replace(',', '').strip())
            upper_value = int(income_values[1].replace(',', '').strip())
            return (lower_value + upper_value) / 2
        else:
            return int(income.replace(',', '').strip())
    except Exception as e:
        return np.nan  # Return NaN if error occurs

# Function to encode categorical columns consistently
def encode_categorical(data, le_dict=None):
    if le_dict is None:
        le_dict = {}
    for column in ['gender', 'region', 'smoke', 'type']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        le_dict[column] = le
    return data, le_dict

# Train the model (this should run at the start of the script)
@st.cache_data
def train_model(data):
    # Clean and encode data
    data['gross_income'] = data['gross_income'].apply(clean_income)
    data, le_dict = encode_categorical(data)

    features = ['age', 'gross_income', 'gender', 'region', 'amt_weekdays', 'amt_weekends']
    target = 'type'

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    return model, le_dict

# Load your dataset (ensure it is available for this part)
data = pd.read_csv("DataSet/smoking.csv")

# Train the model globally
model, label_encoder_dict = train_model(data)

# PAGE 6: Prediction - Updated
if selected_page == "Prediction":
    st.title("🔮 Predict Your Smoking Type")
    st.write("Fill in the details below to predict your smoking type.")

    with st.form(key="prediction_form"):
        age = st.slider("Age", min_value=18, max_value=100, value=30)
        income = st.selectbox("Income", options=data['gross_income'].unique())
        gender = st.selectbox("Gender", options=['Male', 'Female'])  # You can replace with exact values from data
        region = st.selectbox("Region", options=data['region'].unique())
        amt_weekdays = st.slider("Amount Smoked on Weekdays", 0, 10, 0)
        amt_weekends = st.slider("Amount Smoked on Weekends", 0, 10, 0)
        
        submit_button = st.form_submit_button(label="Predict Smoking Type")

    if submit_button:
        # Clean and encode user input
        cleaned_income = clean_income(income)
        gender_encoded = label_encoder_dict['gender'].transform([gender])[0]  # Encode gender input
        region_encoded = label_encoder_dict['region'].transform([region])[0]  # Encode region input
        
        user_input = np.array([[age, cleaned_income, gender_encoded, region_encoded, amt_weekdays, amt_weekends]])

        # Predict smoking type
        prediction = model.predict(user_input)
        predicted_smoking_type = label_encoder_dict['type'].inverse_transform(prediction)

        st.write(f"Prediction: {predicted_smoking_type[0]}")