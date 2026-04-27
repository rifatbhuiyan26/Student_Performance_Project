import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


 


 
# 1. Create output folders
 
FIGURE_FOLDER = "figures"
os.makedirs(FIGURE_FOLDER, exist_ok=True)


 
# 2. Load dataset
 
DATASET_PATH = "StudentsPerformance.csv"

try:
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: {DATASET_PATH} not found.")
    print("Please keep StudentsPerformance.csv in the same folder as this Python file.")
    exit()


 
# 3. Data Understanding
 
print("\n" + "=" * 60)
print("DATA UNDERSTANDING")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())


 
# 4. Data Cleaning
 
print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Keep original row count
original_rows = df.shape[0]

# Cleaning Step 1: Rename columns for easier coding
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("/", "_")
)

print("\nCleaned Column Names:")
print(df.columns.tolist())


# Cleaning Step 2: Remove duplicate rows
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates()

print(f"\nDuplicate rows found: {duplicate_count}")
print(f"Rows after removing duplicates: {df.shape[0]}")


# Cleaning Step 3: Clean categorical text values
categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:
    df[col] = df[col].astype(str).str.strip().str.lower()

print("\nCategorical text values cleaned.")


# Cleaning Step 4: Validate score columns
score_columns = ["math_score", "reading_score", "writing_score"]

for col in score_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

invalid_score_rows = df[
    (df["math_score"] < 0) | (df["math_score"] > 100) |
    (df["reading_score"] < 0) | (df["reading_score"] > 100) |
    (df["writing_score"] < 0) | (df["writing_score"] > 100)
]

print(f"\nInvalid score rows found: {invalid_score_rows.shape[0]}")

# Remove rows with invalid scores
df = df[
    (df["math_score"].between(0, 100)) &
    (df["reading_score"].between(0, 100)) &
    (df["writing_score"].between(0, 100))
]

# Remove rows with missing score values
missing_score_rows = df[score_columns].isnull().any(axis=1).sum()
df = df.dropna(subset=score_columns)

print(f"Rows with missing scores removed: {missing_score_rows}")
print(f"Final rows after cleaning: {df.shape[0]}")


 
# 5. Feature Engineering
 
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Feature 1: Average score
df["average_score"] = df[score_columns].mean(axis=1)

# Feature 2: Performance level
df["performance_level"] = np.where(
    df["average_score"] >= 80,
    "high",
    np.where(df["average_score"] >= 60, "medium", "low")
)

# Feature 3: Pass status
df["pass_status"] = np.where(df["average_score"] >= 40, "pass", "fail")

# Feature 4: Score gap between reading and math
df["reading_math_gap"] = df["reading_score"] - df["math_score"]

print("\nNew Features Created:")
print(["average_score", "performance_level", "pass_status", "reading_math_gap"])

print("\nSample After Feature Engineering:")
print(df.head())


 
# 6. NumPy-Based Custom Computation
 
print("\n" + "=" * 60)
print("NUMPY-BASED CUSTOM COMPUTATION")
print("=" * 60)

# Z-score calculation using NumPy
average_array = df["average_score"].to_numpy()

mean_score = np.mean(average_array)
std_score = np.std(average_array)

df["average_score_z"] = (df["average_score"] - mean_score) / std_score

# Outlier detection using z-score
df["outlier_status"] = np.where(
    np.abs(df["average_score_z"]) > 2,
    "outlier",
    "normal"
)

print(f"\nMean Average Score: {mean_score:.2f}")
print(f"Standard Deviation of Average Score: {std_score:.2f}")
print("\nOutlier Status Count:")
print(df["outlier_status"].value_counts())


 
# 7. Data Analysis
 
print("\n" + "=" * 60)
print("DATA ANALYSIS")
print("=" * 60)

# Analysis 1: Overall score summary
overall_summary = df[["math_score", "reading_score", "writing_score", "average_score"]].agg(
    ["mean", "median", "min", "max", "std"]
)

print("\nAnalysis 1: Overall Score Summary")
print(overall_summary)


# Analysis 2: Gender-wise average score comparison
gender_score = df.groupby("gender")[["math_score", "reading_score", "writing_score", "average_score"]].mean()

print("\nAnalysis 2: Gender-wise Average Score")
print(gender_score)


# Analysis 3: Test preparation course comparison
test_prep_score = df.groupby("test_preparation_course")[["average_score"]].mean().sort_values(
    by="average_score",
    ascending=False
)

print("\nAnalysis 3: Test Preparation Course vs Average Score")
print(test_prep_score)


# Analysis 4: Parental education level comparison
parent_education_score = df.groupby("parental_level_of_education")[["average_score"]].mean().sort_values(
    by="average_score",
    ascending=False
)

print("\nAnalysis 4: Parental Education Level vs Average Score")
print(parent_education_score)


# Analysis 5: Lunch type comparison
lunch_score = df.groupby("lunch")[["average_score"]].mean().sort_values(
    by="average_score",
    ascending=False
)

print("\nAnalysis 5: Lunch Type vs Average Score")
print(lunch_score)


# Analysis 6: Relationship analysis using correlation
correlation_matrix = df[["math_score", "reading_score", "writing_score", "average_score"]].corr()

print("\nAnalysis 6: Correlation Between Scores")
print(correlation_matrix)


# Analysis 7: Performance level distribution
performance_distribution = df["performance_level"].value_counts()

print("\nAnalysis 7: Performance Level Distribution")
print(performance_distribution)


# Analysis 8: Outlier analysis
outliers = df[df["outlier_status"] == "outlier"]

print("\nAnalysis 8: Outlier Students")
print(outliers[["gender", "math_score", "reading_score", "writing_score", "average_score", "average_score_z"]])


 
# 8. Save Analysis Results to Text File
 
with open("analysis_results.txt", "w", encoding="utf-8") as file:
    file.write("Student Performance Analysis Results\n")
    file.write("=" * 60 + "\n\n")

    file.write("Dataset Shape After Cleaning:\n")
    file.write(str(df.shape) + "\n\n")

    file.write("Overall Score Summary:\n")
    file.write(str(overall_summary) + "\n\n")

    file.write("Gender-wise Average Score:\n")
    file.write(str(gender_score) + "\n\n")

    file.write("Test Preparation Course vs Average Score:\n")
    file.write(str(test_prep_score) + "\n\n")

    file.write("Parental Education Level vs Average Score:\n")
    file.write(str(parent_education_score) + "\n\n")

    file.write("Lunch Type vs Average Score:\n")
    file.write(str(lunch_score) + "\n\n")

    file.write("Correlation Between Scores:\n")
    file.write(str(correlation_matrix) + "\n\n")

    file.write("Performance Level Distribution:\n")
    file.write(str(performance_distribution) + "\n\n")

    file.write("Outlier Students:\n")
    file.write(str(outliers[["gender", "math_score", "reading_score", "writing_score", "average_score", "average_score_z"]]) + "\n\n")

print("\nAnalysis results saved as analysis_results.txt")


 
# 9. Visualization using Matplotlib
 

# Chart 1: Test preparation course vs average score
plt.figure(figsize=(8, 5))
plt.bar(test_prep_score.index, test_prep_score["average_score"])
plt.title("Average Score by Test Preparation Course")
plt.xlabel("Test Preparation Course")
plt.ylabel("Average Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{FIGURE_FOLDER}/chart_1_test_preparation_vs_average_score.png", dpi=300)
plt.show()


# Chart 2: Parental education level vs average score
plt.figure(figsize=(10, 6))
plt.bar(parent_education_score.index, parent_education_score["average_score"])
plt.title("Average Score by Parental Level of Education")
plt.xlabel("Parental Level of Education")
plt.ylabel("Average Score")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(f"{FIGURE_FOLDER}/chart_2_parental_education_vs_average_score.png", dpi=300)
plt.show()


# Chart 3: Reading score vs writing score scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(df["reading_score"], df["writing_score"], alpha=0.6)
plt.title("Relationship Between Reading Score and Writing Score")
plt.xlabel("Reading Score")
plt.ylabel("Writing Score")
plt.tight_layout()
plt.savefig(f"{FIGURE_FOLDER}/chart_3_reading_vs_writing_score.png", dpi=300)
plt.show()


# Chart 4: Average score distribution histogram
plt.figure(figsize=(8, 5))
plt.hist(df["average_score"], bins=15, edgecolor="black")
plt.title("Distribution of Average Scores")
plt.xlabel("Average Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(f"{FIGURE_FOLDER}/chart_4_average_score_distribution.png", dpi=300)
plt.show()


# Chart 5: Box plot of math score by gender
gender_groups = [group["math_score"].values for name, group in df.groupby("gender")]
gender_labels = [name for name, group in df.groupby("gender")]

plt.figure(figsize=(8, 5))
plt.boxplot(gender_groups, labels=gender_labels)
plt.title("Math Score Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Math Score")
plt.tight_layout()
plt.savefig(f"{FIGURE_FOLDER}/chart_5_math_score_by_gender_boxplot.png", dpi=300)
plt.show()


# Chart 6: Performance level distribution
plt.figure(figsize=(8, 5))
plt.bar(performance_distribution.index, performance_distribution.values)
plt.title("Performance Level Distribution")
plt.xlabel("Performance Level")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(f"{FIGURE_FOLDER}/chart_6_performance_level_distribution.png", dpi=300)
plt.show()


 
# 10. Save Cleaned Dataset
 
df.to_csv("cleaned_students_performance.csv", index=False)

print("\nCleaned dataset saved as cleaned_students_performance.csv")
print(f"All charts saved inside the '{FIGURE_FOLDER}' folder.")
print("\nProject completed successfully!")