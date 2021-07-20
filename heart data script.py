import os

# check working directory
os.getcwd()
os.chdir(r"/users/aineodonoghue/downloads")
os.getcwd()


import pandas as pd

hd = pd.read_csv("heart_failure_clinical_records_dataset.csv")
# checking shape of dataset
hd.shape
print(hd.shape)

# checking top 5 rows
print(hd.head())
# Check column names
print(hd.columns)
# check description of all data points
print(hd.describe().T)
# checking the data types of the dataset
print(hd.dtypes)
# all data types are ints and floats

# clean dataset
# check for null values and duplicates in the dataset
missing_values_count = hd.isnull().sum()
print(missing_values_count)
# no missing values (if there was could use mean, median, mode for example)
# Select duplicate rows except first occurrence based on all columns
duplicateRowsDF =hd[hd.duplicated()]
print("Duplicate Rows except first occurrence based on all columns are :")
print(duplicateRowsDF)
# no duplicates identified

# analysis dataset
print(hd["DEATH_EVENT"].value_counts())
# check total output variable counts, 1 means the person has died, 0 they have survived
# 203 people survived and 96 died, the dataset is imbalanced, will need to look at relevant variables closer
print(hd["sex"].value_counts())
# 1 = male and 0 = female, there are 194 men to 105 women, biased towards men

# import visualisation libs
import seaborn as sns
import matplotlib.pyplot as plt

df2 = hd.copy()

sns.countplot(x="DEATH_EVENT", hue="sex", data=df2)

def chng(sex):
    if sex == 0:
        return 'female'
    else:
        return 'male'

def chng2(DEATH_EVENT):
    if DEATH_EVENT == 0:
        return "survived"
    else:
        return "died"

df2['sex'] = df2['sex'].apply(chng)
df2['DEATH_EVENT'] = df2['DEATH_EVENT'].apply(chng2)
# looks like similiar stats for both men and women given the sample, approx 60-70% survive, 30-40% died.

x1 = hd.sex
y1 = hd.DEATH_EVENT

from scipy.stats import pearsonr
corr, x=pearsonr(x1,y1)
print(corr)
# corr are close to zero

plt.figure(figsize=(12,10))
sns.heatmap(hd.corr(),annot=True,cmap="magma",fmt='.2f')

# heatmap to show the corr between variables
# ejection fraction, serum creatinine and time and age look like the most impactful
# the sex shows little correlation, which is in line with the previous analysis
# with the imbalance in the dataset, let's look at the most relevant features




