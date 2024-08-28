import json
import pandas as pd
import dateutil.parser
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1: Load the JSON data
with open('C:/Users/msi/Desktop/cve.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Step 2: Normalize the nested JSON data and load it into a DataFrame
df = pd.json_normalize(data['CVE_Items'])

# Step 3: Explore the DataFrame
print("Initial Data Exploration:")
print(df.head())
print(df.info())
print(df.describe())

# Step 4: Data Cleaning
# Handle missing values: Drop rows with missing critical data
df_cleaned = df.dropna(subset=['impact.baseMetricV3.cvssV3.baseScore']).copy()

# Convert date columns to datetime using .apply, handling errors
def parse_date(date_str):
    try:
        return dateutil.parser.parse(date_str)
    except ValueError:
        return pd.NaT

df_cleaned['publishedDate'] = df_cleaned['publishedDate'].apply(parse_date)
df_cleaned['lastModifiedDate'] = df_cleaned['lastModifiedDate'].apply(parse_date)

# Verify the conversion
print("Data types after conversion:")
print(df_cleaned['publishedDate'].dtype)
print(df_cleaned['lastModifiedDate'].dtype)

# Inspect the first few rows to check for any issues with conversion
print(df_cleaned[['publishedDate', 'lastModifiedDate']].head())

# Drop rows with non-datetime values
df_cleaned = df_cleaned[(df_cleaned['publishedDate'].notnull()) & (df_cleaned['lastModifiedDate'].notnull())]

# Step 5: Feature Engineering
# Create a new feature for the time difference between publishedDate and lastModifiedDate
df_cleaned['time_to_modify'] = (df_cleaned['lastModifiedDate'] - df_cleaned['publishedDate']).dt.days

# Categorize severity based on CVSS base scores
df_cleaned['severity_category'] = pd.cut(df_cleaned['impact.baseMetricV3.cvssV3.baseScore'],
                                         bins=[0, 3.9, 6.9, 8.9, 10],
                                         labels=['Low', 'Medium', 'High', 'Critical'])

# Step 6: Save the Processed Data
df_cleaned.to_csv('C:/Users/msi/Desktop/cve_data_processed.csv', index=False)
print("Data processing complete. The processed file is saved as 'cve_data_processed.csv'.")

# Step 7: Exploratory Data Analysis (Optional)

# Filter the DataFrame to only include numeric columns
numeric_df = df_cleaned.select_dtypes(include=['float64', 'int64'])

# Example: Correlation Matrix
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()

# Example: Distribution of CVSS Base Scores
sns.histplot(df_cleaned['impact.baseMetricV3.cvssV3.baseScore'], bins=10, kde=True)
plt.title('Distribution of CVSS Base Scores')
plt.xlabel('CVSS Base Score')
plt.ylabel('Frequency')
plt.show()

# Step 8: Predictive Modeling (Optional)
# Example: Simple Random Forest Classifier
X = numeric_df[['impact.baseMetricV3.cvssV3.baseScore', 'time_to_modify']]
y = df_cleaned['severity_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
