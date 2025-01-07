# Analysis of Smoking Behavior (UK Dataset)  

## Semester Project  
### **Introduction to Data Science**  

This project focuses on analyzing smoking habits using a dataset from the UK. The analysis provides insights into smoking behavior based on various factors, including gender, age, income, region, and more. The interactive dashboard is built using **Streamlit**, and it employs advanced data cleaning, machine learning, and visualization techniques for meaningful insights.  

---

## **Features of the Project**  

### **Page Configuration:**  
- Designed using Streamlit with custom page settings (title, icon, and layout).  

### **Data Loading and Cleaning:**  
- **Data Loading:** The dataset is loaded using Pandas.  
- **Data Cleaning:** Missing values are handled and imputed using median/mode for smoking-related columns, enhancing data quality for analysis.  

### **Machine Learning Integration:**  
- A **Random Forest Classifier** is used to predict smoking type based on user inputs.  
- Supports interactive predictions through the dashboard's "Prediction" page.  

### **Sidebar Navigation:**  
- A dynamic sidebar allows seamless navigation across different sections of the analysis.  

---

## **Pages Overview**  

### **Overview & Demographics**  
- **Total Records:** Displays the total number of records in the dataset.  
- **Smokers & Non-Smokers:** Summarizes smoking trends with metrics and percentages.  
- **Gender-Based Smoking Trends:** Interactive bar chart showing smoking behavior by gender.  
- **Average Age:** Calculates and displays the average age of individuals.  

---

### **Smoking Behavior**  
- Explores smoking habits based on income, education, and smoking type:  
  - **Income Analysis:** Grouped bar charts visualizing smoking trends by income bracket.  
  - **Education Analysis:** Bar chart representation of smoking trends across educational qualifications.  
  - **Smoking Type Distribution:** Pie chart showcasing the distribution of different smoking types.  

---

### **Trends & Correlations**  
- **Age Distribution:** Histograms analyzing age-based smoking behavior.  
- **Smoking Frequency:** Scatter plots comparing smoking on weekdays vs. weekends.  
- **Correlation Analysis:** Heatmap to visualize relationships between numerical features.  

---

### **Advanced Analysis**  
- **Age Group Trends:** Highlights smoking behavior across categorized age groups.  
- **Regional Patterns:** Stacked bar charts showcasing regional smoking trends.  
- **Marital Status:** Explores smoking behavior trends by marital status.  

---

### **Income and Smoking Type**  
- Examines the relationship between gross income and smoking type.  
- Interactive bar chart shows the distribution of smoking types across income brackets.  

---

### **Prediction**  
- A dedicated page to predict smoking type based on user inputs, leveraging a pre-trained **Random Forest Classifier**:  
  - Inputs include age, income, gender, region, and smoking frequency on weekdays/weekends.  
  - Outputs the predicted smoking type.  

---

## **Technologies Used**  
- **Python:** Programming language for data processing and analysis.  
- **Streamlit:** Framework for building the interactive web application.  
- **Pandas:** Data manipulation and cleaning.  
- **Seaborn & Matplotlib:** For advanced visualizations and plots.  
- **Plotly Express:** Interactive and dynamic data visualizations.  
- **Scikit-learn:** Machine learning model implementation for predictions.  

---

## **How to Run the Project**  
1. Clone the repository and ensure the dataset (`smoking.csv`) is in the `DataSet` folder.  
2. Install the required Python libraries:  
   ```bash
   pip install streamlit pandas plotly seaborn matplotlib scikit-learn
