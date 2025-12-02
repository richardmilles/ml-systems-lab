# Employee Attrition Prediction

## 1. Introduction

### 1.1 Project Context
Employee attrition is a major issue in many companies. High staff turnover can lead to significant costs in recruitment, training, and loss of skills. Understanding the reasons for voluntary departures and predicting which employees are likely to leave would allow companies to anticipate these departures and take corrective measures.

### 1.2 Project Objectives
The main objective of this project is to develop a machine learning model capable of predicting whether an employee is likely to leave the company. More specifically, we aim to:
- Identify the most influential factors on employee attrition.
- Build an accurate and robust predictive model.
- Provide recommendations to HR teams to limit attrition.

## 2. Data Exploration and Analysis

### 2.1 Dataset Overview
The dataset used comes from a fictional company and contains 1480 observations and 38 variables. These variables cover various aspects of employment, including:
- Demographic information: Age, gender, marital status.
- Professional characteristics: Job role, department, tenure.
- Working conditions: Salary, overtime, job satisfaction.
- Target variable: Attrition (Yes/No), indicating if the employee has left the company.

### 2.2 Exploratory Analysis
Before building a model, it is essential to explore the data to detect trends and anomalies.
- Global attrition rate: About 16% of employees leave the company.
- Influence of overtime: A large portion of employees who worked overtime left the company.
- Impact of salary: Employees with lower salaries seem more inclined to leave.
- Tenure and departures: A significant proportion of employees leave after a few years in the company.

## 3. Modeling and Experiments

### 3.1 Tested Models
We tested three different models to compare their performance:

- Logistic Regression: Global Accuracy 85%, Recall (class 1) 23%, Precision (class 1) 61%
- Random Forest: Global Accuracy 84%, Recall (class 1) 49%, Precision (class 1) 50%
- XGBoost: Global Accuracy 83%, Recall (class 1) 57%, Precision (class 1) 47%

### 3.2 Choice of Best Model
The XGBoost model was selected because it offers the best compromise between precision and recall. Its recall of 57% means that it detects more than half of the employees who will leave the company, which is a satisfactory result compared to the other models.

## 4. Key Attrition Factor Analysis

### 4.1 Most Important Variables
According to our model, the variables with the most impact on attrition are:
1. Overtime (OverTime): Excessive workload is a departure factor.
2. Marital Status - Single: Single employees are more inclined to change jobs.
3. Job Role: Sales representatives and production directors are more at risk.
4. Tenure (YearsAtCompany): A critical threshold seems to exist beyond which employees leave.
5. Job Satisfaction (JobSatisfaction): A low level of satisfaction increases the risk of departure.

## 5. Recommendations and Practical Applications

### 5.1 Recommended HR Actions
1. Monitor overtime and implement policies to reduce workload.
2. Offer clear career prospects for at-risk employees.
3. Review working conditions for sales representatives and production directors.
4. Improve job satisfaction by adapting compensation and benefits.
5. Individual monitoring of employees with high attrition risk via regular interviews.

## 6. Model Implementation and Usage

The final model is saved for future predictions:

A user interface based on Streamlit or Flask could be developed to allow for more intuitive use by HR teams.

## 7. Conclusion and Future Outlook

This project allowed for the construction of a decision support tool to anticipate employee attrition. The results show that departures can be partially predicted, thus allowing companies to take preventive measures.

### Perspectives for Improvement
- Exploration of other more complex models like LightGBM.
- Integration of new variables such as performance reviews.
- Implementation of a real-time tracking system to proactively anticipate departures.
