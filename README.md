# Analysis of Smoking Behavior (UK Dataset)  

## Semester Project  
### **Introduction to Data Science**  

This project focuses on analyzing smoking habits using a dataset from the UK. The analysis provides insights into smoking behavior based on various factors, including gender, age, income, region, and more. The interactive dashboard is built using **Streamlit**, and it employs data cleaning and visualization techniques for meaningful insights.  

---

## **Features of the Project**  

### **Page Configuration:**  
- Designed using Streamlit with custom page settings (title, icon, and layout).  

### **Data Loading and Cleaning:**  
- **Data Loading:** The dataset is loaded using Pandas.  
- **Data Cleaning:** Missing values are handled, and the dataset is enriched with imputed values (median/mode) for smoking-related columns.  

### **Sidebar Navigation:**  
- A dynamic sidebar allows seamless navigation across different sections of the analysis.  

---

## **Pages Overview**  

### **Overview Page**  
- **Total Records:** Displays the total number of records in the dataset.  
- **Smokers:** Number of individuals who smoke.  
- **Non-Smokers:** Number of individuals who don't smoke.  
- **Average Age:** Shows the average age of all individuals.  
- **Proportion of Smokers and Non-Smokers:** Pie chart visualizing the proportion of smokers vs. non-smokers.  

---

### **Smoking by Gender**  
- Provides insights into smoking behavior categorized by gender.  
- Includes a grouped bar chart to compare smoking habits between males and females.  

---

### **Age Distribution**  
- Explores how smoking behavior varies across different age groups.  
- Visualized using a histogram with KDE (Kernel Density Estimation) overlay.  

---

### **Income Analysis**  
- Examines the correlation between gross income and smoking habits.  
- Displays smoking trends across various income brackets using a grouped bar chart.  

---

### **Smoking Type Distribution**  
- Analyzes the distribution of different smoking types (e.g., light, moderate, heavy).  
- Presented as a pie chart for a clear understanding of proportions.  

---

### **Weekday vs Weekend Smoking**  
- Compares smoking frequency on weekdays vs. weekends.  
- Scatter plot showcasing relationships between weekday and weekend smoking habits.  

---

### **Regional Smoking Patterns**  
- Investigates smoking patterns across different UK regions.  
- Visualized using a stacked bar chart to highlight proportions in each region.  

---

### **Correlation Analysis**  
- Examines correlations between numerical features in the dataset.  
- Heatmap is used to display correlation coefficients and relationships visually.  

---

### **Smoking by Education**  
- Studies smoking habits based on educational qualifications.  
- Bar chart representation of smoking trends across education levels.  

---

### **Smoking Frequency Comparison**  
- Compares average cigarette consumption on weekdays and weekends.  
- Metrics and a bar chart illustrate the differences in smoking frequency.  

---

### **Smoking by Ethnicity**  
- Highlights smoking behavior trends across different ethnic groups.  
- Grouped bar chart is used to compare smoking habits within ethnicities.  

---

## **Technologies Used**  
- **Python:** Programming language for data processing and analysis.  
- **Streamlit:** Framework for building the interactive web application.  
- **Pandas:** Data manipulation and cleaning.  
- **Seaborn & Matplotlib:** For advanced visualizations and plots.  
- **Plotly Express:** Interactive and dynamic data visualizations.  

---

### **How to Run the Project**  
1. Clone the repository and ensure the dataset (`smoking.csv`) is in the `DataSet` folder.  
2. Install the required Python libraries:  
   ```bash
   pip install streamlit pandas plotly seaborn matplotlib
