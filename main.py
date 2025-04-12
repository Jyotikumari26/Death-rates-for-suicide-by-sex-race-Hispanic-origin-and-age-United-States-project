import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"C:\Users\hp\Downloads\Death_rates_for_suicide__by_sex__race__Hispanic_origin__and_age__United_States (1).csv")

# Basic Data Exploration
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# Clean and prepare the data
# Filter only age-adjusted rates for consistency
df = df[df['UNIT'] == 'Deaths per 100,000 resident population, age-adjusted']

# Drop unnecessary columns
df = df.drop(['UNIT_NUM', 'STUB_NAME_NUM', 'STUB_LABEL_NUM', 'YEAR_NUM', 'AGE_NUM', 'FLAG'], axis=1, errors='ignore')

# Convert ESTIMATE to numeric (some values might be missing)
df['ESTIMATE'] = pd.to_numeric(df['ESTIMATE'], errors='coerce')

# Basic statistics
print("\nBasic statistics:")
print(df['ESTIMATE'].describe())

# 1. Overall Trend Analysis
plt.figure(figsize=(14, 6))
overall = df[df['STUB_LABEL'] == 'All persons'].dropna(subset=['ESTIMATE'])
sns.lineplot(data=overall, x='YEAR', y='ESTIMATE', marker='o')
plt.title('Overall Suicide Rate Trend (1950-2018)')
plt.xlabel('Year')
plt.ylabel('Deaths per 100,000 (age-adjusted)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Gender Comparison
gender_data = df[df['STUB_NAME'] == 'Sex'].dropna(subset=['ESTIMATE'])
plt.figure(figsize=(14, 6))
sns.lineplot(data=gender_data, x='YEAR', y='ESTIMATE', hue='STUB_LABEL', marker='o')
plt.title('Suicide Rates by Gender (1950-2018)')
plt.xlabel('Year')
plt.ylabel('Deaths per 100,000 (age-adjusted)')
plt.legend(title='Gender')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Race Analysis (Male)
male_race = df[df['STUB_NAME'].str.contains('Sex and race') & 
              df['STUB_LABEL'].str.startswith('Male:')].dropna(subset=['ESTIMATE'])
plt.figure(figsize=(16, 8))
sns.lineplot(data=male_race, x='YEAR', y='ESTIMATE', hue='STUB_LABEL', marker='o')
plt.title('Male Suicide Rates by Race (1950-2018)')
plt.xlabel('Year')
plt.ylabel('Deaths per 100,000 (age-adjusted)')
plt.legend(title='Race', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Race Analysis (Female)
female_race = df[df['STUB_NAME'].str.contains('Sex and race') & 
               df['STUB_LABEL'].str.startswith('Female:')].dropna(subset=['ESTIMATE'])
plt.figure(figsize=(16, 8))
sns.lineplot(data=female_race, x='YEAR', y='ESTIMATE', hue='STUB_LABEL', marker='o')
plt.title('Female Suicide Rates by Race (1950-2018)')
plt.xlabel('Year')
plt.ylabel('Deaths per 100,000 (age-adjusted)')
plt.legend(title='Race', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Recent Year Comparison (2018)
recent_data = df[df['YEAR'] == 2018].dropna(subset=['ESTIMATE'])
recent_data = recent_data[recent_data['STUB_NAME'].isin(['Sex', 'Sex and race', 'Sex and race and Hispanic origin'])]

plt.figure(figsize=(16, 8))
sns.barplot(data=recent_data, y='STUB_LABEL', x='ESTIMATE', palette='viridis')
plt.title('Suicide Rates by Demographic Group (2018)')
plt.xlabel('Deaths per 100,000 (age-adjusted)')
plt.ylabel('Demographic Group')
plt.tight_layout()
plt.show()

# 6. Age Group Analysis (Crude rates)
age_data = df[df['STUB_NAME'] == 'Age'].dropna(subset=['ESTIMATE'])
age_data = age_data[age_data['AGE'].isin(['10-14 years', '15-24 years'])]

plt.figure(figsize=(14, 6))
sns.lineplot(data=age_data, x='YEAR', y='ESTIMATE', hue='AGE', style='AGE', 
             markers=True, dashes=False)
plt.title('Suicide Rates by Age Group (1950-2018)')
plt.xlabel('Year')
plt.ylabel('Deaths per 100,000 (crude)')
plt.legend(title='Age Group')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7. Hispanic Origin Analysis
hispanic_data = df[df['STUB_NAME'] == 'Sex and race and Hispanic origin'].dropna(subset=['ESTIMATE'])
hispanic_data = hispanic_data[hispanic_data['STUB_LABEL'].str.contains('Hispanic')]

plt.figure(figsize=(16, 8))
sns.lineplot(data=hispanic_data, x='YEAR', y='ESTIMATE', hue='STUB_LABEL', style='STUB_LABEL')
plt.title('Suicide Rates by Hispanic Origin (1950-2018)')
plt.xlabel('Year')
plt.ylabel('Deaths per 100,000 (age-adjusted)')
plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 8. Boxplot of Suicide Rates by Demographic Groups
# Select recent data for better visualization
recent_years = df[df['YEAR'] >= 2000].dropna(subset=['ESTIMATE'])
recent_years = recent_years[recent_years['STUB_NAME'].isin(['Sex', 'Sex and race'])]

plt.figure(figsize=(16, 10))
sns.boxplot(data=recent_years, y='STUB_LABEL', x='ESTIMATE', palette='Set2')
plt.title('Distribution of Suicide Rates by Demographic Groups (2000-2018)')
plt.xlabel('Deaths per 100,000 (age-adjusted)')
plt.ylabel('Demographic Group')
plt.tight_layout()
plt.show()

# 9. Correlation between male and female rates by year
gender_pivot = gender_data.pivot_table(index='YEAR', columns='STUB_LABEL', values='ESTIMATE')
gender_pivot = gender_pivot.dropna()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=gender_pivot, x='Male', y='Female')
plt.title('Correlation Between Male and Female Suicide Rates')
plt.xlabel('Male Suicide Rate')
plt.ylabel('Female Suicide Rate')

# Add correlation coefficient
corr = gender_pivot.corr().iloc[0,1]
plt.text(0.1, 0.9, f'Pearson r = {corr:.2f}', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

# 10. Heatmap of suicide rates by year and demographic group
heatmap_data = recent_years.pivot_table(index='STUB_LABEL', columns='YEAR', values='ESTIMATE')

plt.figure(figsize=(16, 12))
sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.1f', linewidths=.5)
plt.title('Suicide Rates Heatmap by Demographic Group and Year')
plt.xlabel('Year')
plt.ylabel('Demographic Group')
plt.tight_layout()
plt.show()
