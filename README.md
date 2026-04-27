# Student Performance Analysis Project

## Project Overview

This project is a data analysis project based on the **Students Performance Dataset**. The main goal of this project is to analyze how different factors such as gender, parental education level, lunch type, and test preparation course are related to student exam performance.

The project uses **Pandas**, **NumPy**, and **Matplotlib** to clean the data, create new features, perform meaningful analysis, and generate visualizations.

---

## Student Information

**Student Name:** Rifat Bhuiyan  
**Student ID:** 22-49356-3  
**Course:** Programming in Python  
**Project Type:** Final Term Data Analysis Project  

---

## Dataset Information

**Dataset Name:** Students Performance Dataset  
**Dataset File:** `StudentsPerformance.csv`  
**Dataset Source:** Kaggle  
**Number of Rows:** 1000  
**Number of Columns:** 8  

Each row represents one student record. The dataset contains student demographic and academic information.

### Main Columns

- Gender
- Race / Ethnicity
- Parental Level of Education
- Lunch
- Test Preparation Course
- Math Score
- Reading Score
- Writing Score

---

## Project Objectives

The main objectives of this project are:

1. To understand the overall performance pattern of students.
2. To compare student performance based on gender.
3. To analyze whether test preparation course completion affects average score.
4. To examine the relationship between parental education level and student performance.
5. To identify score relationships and unusual performance patterns.

---

## Research Questions

This project answers the following analytical questions:

1. Do students who completed the test preparation course perform better on average?
2. Does parental level of education show any difference in student performance?
3. What is the relationship between math, reading, and writing scores?
4. Are there any unusual or outlier performance records in the dataset?

---

## Tools and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- VS Code
- Git and GitHub

---

## Project Structure

```text
Student_Performance_Project/
│
├── StudentsPerformance.csv
├── student_analysis.py
├── cleaned_students_performance.csv
├── analysis_results.txt
├── figures/
│   ├── chart_1_test_preparation_vs_average_score.png
│   ├── chart_2_parental_education_vs_average_score.png
│   ├── chart_3_reading_vs_writing_score.png
│   ├── chart_4_average_score_distribution.png
│   ├── chart_5_math_score_by_gender_boxplot.png
│   └── chart_6_performance_level_distribution.png
└── README.md
