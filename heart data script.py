import os
import pandas as pd
import numpy as np
os.getcwd()
# check working directory
os.chdir(r"/users/aineodonoghue/downloads")
# change working directory to where files are saved
os.getcwd()
# check new working directory location has changed to correct file
hd = pd.read_csv("heart_failure_clinical_records_dataset.csv")
# import csv file using pandas
hd.head()
# check what the data looks like, top 5 rows
hd.shape
# check shape of dataframe
hd.columns
# check column headings
hd.dtypes
# check data set for date types
# all numerical values in floats and integers
hd.describe().T
# mean, std, and distribution of dataset

import re
## regex function to see find alphanumeric characters after the word User in the text
re.findall(r"User\w","The winners are:User9,User678,User8,UserN, UserNRT")
## regex function to see find all alphanumeric characters after the word User in text
## the + sign means all alphanumeric characters after User not just first character as above
re.findall(r"User\w+","The winners are:User9,User678,User8,UserN, UserNRT")

## Example of concatenation of string and integer data type by converting int to string
print("How old are you. " + "I am " + str(40) + " years old")
"""
Two dataframes can be concatenated using pandas function pd.concat. 
It can do if on the horizontal or vertical axis using axis 0 or 1. 
I only have one dataset so i have no need to join, merge or concatenate.
"""

[print(col) for col in hd if hd[col].isna().sum() > 0]
# no null values

drop_duplicates= hd.drop_duplicates()
print(hd.shape,drop_duplicates.shape)
# (299,13) before and after, no duplicates found

hd["DEATH_EVENT"].value_counts()
# check total output variable counts

hd["sex"].value_counts()
#  check total gender distribution  counts

#import visualisation libs
import seaborn as sns
import matplotlib.pyplot as plt

df2 = hd.copy()


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

sns.countplot(x="DEATH_EVENT", hue="sex", data=df2)

# looks like similiar stats for both men and women given the sample, approx 60-70% survive.

df2 = hd.copy()


def chng3(smoking):
    if smoking == 0:
        return 'non smoker'
    else:
        return 'smoker'


df2['smoking'] = df2['smoking'].apply(chng3)
df2['DEATH_EVENT'] = df2['DEATH_EVENT'].apply(chng2)

sns.countplot(x="DEATH_EVENT", hue="smoking", data=df2)

# again as with sex the death ratio is approx the same for both in the 30%
# let's run the corr heatmap to check the correlations bwt the variables

cols=["#4AA96C", "#F55C47"]
sns.relplot(data=hd, x="age", y="sex", hue="DEATH_EVENT", palette=cols)
cols=["#4AA96C", "#F55C47"]
sns.relplot(data=hd, x="time", y="sex", hue="DEATH_EVENT", palette=cols)
cols=["#4AA96C", "#F55C47"]
sns.relplot(data=hd, x="age", y="ejection_fraction", hue="DEATH_EVENT", palette=cols)
cols=["#4AA96C", "#F55C47"]
sns.relplot(data=hd, x="age", y="serum_creatinine", hue="DEATH_EVENT", palette=cols)
cols=["#4AA96C", "#F55C47"]
sns.relplot(data=hd, x="age", y="serum_sodium", hue="DEATH_EVENT", palette=cols)
# plotting variables to get more insight

# Feature Selection, checking to see what the most important feature in the dataset are before selection, if any.

plt.rcParams['figure.figsize']=15,6
sns.set_style("darkgrid")

x = hd.iloc[:, :-1]
y = hd.iloc[:,-1]

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()

# age distribution in dataset and against death rate
import plotly.express as px
age_counts = hd["age"].value_counts()
fig = px.bar(age_counts, title="Age distribution")
fig.update_layout(
    xaxis_title = "Age",
    yaxis_title = "Frequency",
    title_x = 0.5,
    showlegend = False
)
fig.show()

age = pd.cut(hd['age'], 8)
fig, axs = plt.subplots(figsize=(15, 8))
sns.countplot(x=age,hue='DEATH_EVENT',
              data=hd,palette=["cornflowerblue", "khaki"]).set_title("Age distrubation with deaths",
                                                                { 'fontsize': 20, 'fontweight':'bold'});
axs.legend(["Didnt Survive","Survived"],
              loc="upper right")

# heatmap to show the corr between variables
# ejection fraction, serum creatinine and time show the most impact
# the sex and diabetes show little correlation, which is in line with the previous graphs
plt.figure(figsize=(12,10))
sns.heatmap(hd.corr(),annot=True,cmap="magma",fmt='.2f')

# investigating outliers in the variables selected for machine learning
# two outliers in ejection fraction
sns.boxplot(x = hd.ejection_fraction, color = 'teal')
plt.show()
#no outliers for time
sns.boxplot(x = hd.age, color = 'teal')
plt.show()
#numerous outliers, however the higher ranges directly affect heart failure so should not be considered outliers
sns.boxplot(x = hd.serum_creatinine, color = 'teal')
plt.show()


# removing outliers from ejection fraction
hd[hd['ejection_fraction']>=70]

#removing ourliers from dataset using index of rows
hd.drop([64,217], axis=0, inplace=True)

#checking shape again as previously 299 rows
hd.shape

#checking boxplot again
sns.boxplot(x = hd.ejection_fraction, color = 'teal')
plt.show()

hd.head()

#dropping columns not needed for model testing as within dataset not shown to have high impact.
hd.drop('creatinine_phosphokinase', axis=1, inplace=True)
hd.drop('high_blood_pressure', axis=1, inplace=True)
hd.drop('platelets', axis=1, inplace=True)
hd.drop('sex', axis=1, inplace=True)
hd.drop('smoking', axis=1, inplace=True)
hd.drop('anaemia', axis =1, inplace=True)
hd.drop('diabetes', axis=1, inplace=True)
hd.drop('time', axis=1, inplace=True)
hd.drop('serum_sodium', axis=1, inplace=True)

# checking header after dropping columns
hd.head()


# check heading to ensure only 4 X variables and 1 y variable in dataset
# all numerical so no need for one hot encoding
hd.head()
# importing SKlearn for modeling
from sklearn.model_selection import train_test_split

#creating different sets of independent and dependent variables
hd_x=hd.drop("DEATH_EVENT", axis=1)
hd_y=hd["DEATH_EVENT"]

# checking shape
hd_x.shape
hd_y.shape

# splitting dataset into test and train
hd_x_train, hd_x_test, hd_y_train, hd_y_test = train_test_split(hd_x, hd_y, test_size=.25, random_state=0)

# checking shape of test and train
hd_x_train.shape
hd_y_train.shape
hd_x_test.shape
hd_y_test.shape

# run logistic regression as this is a binary classification
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
#fit model
logisticRegr.fit(hd_x_train, hd_y_train)
#test data used to evaluate model
predictions = logisticRegr.predict(hd_x_test)
# accuracy score
mylist = []
score = logisticRegr.score(hd_x_test, hd_y_test)
mylist.append(score)
print(score)

from sklearn import metrics
# Creating the confusion matrix
cm = metrics.confusion_matrix(hd_y_test, predictions)
print(cm)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt=" .3f", linewidths=.5, square=True, cmap="Blues_r");
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
# in column Death events, 1 means the person died and 0 means the person survived
# model is looking to predict a death event
#from the figure about the actuals in total were, 52 people survived and 23 people died.
#the model correctly predicted 48 people survived but falsely predicted 4 died when in reality they survived
#the model correctly predicted 18 people died but falsely predicted 5 lived when in reality they died

t_neg= cm[0][0]
f_pos = cm[0][1]
f_neg = cm[1][0]
t_pos = cm[1][1]

accuracy = (t_pos + t_neg)/(t_neg + t_pos + f_neg + f_pos)
accuracy

misclass_rate = (f_pos + f_neg)/(t_neg + t_pos + f_neg + f_pos)
misclass_rate

precision = (t_pos)/(t_pos + f_pos)
precision

recall = (t_pos)/(t_pos + f_neg)
recall

f1_score = 2*(recall*precision)/(recall+precision)
f1_score

# decision tree machine learning
from sklearn import tree

## create tree object
clf = tree.DecisionTreeClassifier(criterion='gini',min_samples_leaf=20)
## fitting the model
clf = clf.fit(hd_x_train, hd_y_train)
## training dataset is only used for training and test dataset should be used to evaluate the performance of model
predictions = clf.predict(hd_x_test)
# Use score method to get accuracy of model
## this score is accuracy
score = clf.score(hd_x_test, hd_y_test)
mylist.append(score)
print(score)
cm = metrics.confusion_matrix(hd_y_test, predictions)
print(cm)
## in column survived, 1 means the person has died and 0 means the person survived
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)

# checking most important features in the model
Feat_Imp = pd.DataFrame({"Features" : hd_x_train.columns,"Importance" : clf.feature_importances_})
Feat_Imp = Feat_Imp.sort_values(by=['Importance'],ascending=False)
print(Feat_Imp)
#
loca = sns.barplot(x="Importance",y="Features",data = Feat_Imp)
plt.show()

"""
In the above figure we can see that, 10 people did not survive and our model also predicted the same
45 people actually survived and our model also predicted the same
13 people did not survive actually, but our model said that they lived
7 people survived actually, but our model said that they had a death event
"""

t_neg= cm[0][0]
f_pos = cm[0][1]
f_neg = cm[1][0]
t_pos = cm[1][1]

accuracy = (t_pos + t_neg)/(t_neg + t_pos + f_neg + f_pos)
accuracy

misclass_rate = (f_pos + f_neg)/(t_neg + t_pos + f_neg + f_pos)
misclass_rate

precision = (t_pos)/(t_pos + f_pos)
precision

# Recall
recall = (t_pos)/(t_pos + f_neg)
recall

f1_score = 2*(recall*precision)/(recall+precision)
f1_score

# Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

## create tree object
clf = RandomForestClassifier()
## fitting the model
clf = clf.fit(hd_x_train, hd_y_train)
# number of trees used
print('Number of Trees used : ', clf.n_estimators)
## training dataset is only used for training and test dataset should be used to evaluate the performance of model
predictions = clf.predict(hd_x_test)
print('\nTarget on test data',predictions)
# Use score method to get accuracy of model
## this score is accuracy
score = clf.score(hd_x_test, hd_y_test)
mylist.append(score)
print(score)
cm = metrics.confusion_matrix(hd_y_test, predictions)
print(cm)
## in column survived, 1 means the person has died and 0 means the person survived
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
# most important feature in random forest
Feat_Imp = pd.DataFrame({"Features" : hd_x_train.columns,"Importance" : clf.feature_importances_})
Feat_Imp = Feat_Imp.sort_values(by=['Importance'],ascending=False)
print(Feat_Imp)
#
loca = sns.barplot(x="Importance",y="Features",data = Feat_Imp)
plt.show()
"""
In the above figure we can see that, 11 people actually did not survive and our model also predicted the same
43 people actually survived and our model also predicted the same
12 people did not survive actually, but our model said that they lived
9 people survived actually, but our model said that they died
"""
true_negative = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_positive = cm[1][1]

Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy
misclassification_rate = (false_positive + false_negative) / (true_positive +false_positive + false_negative + true_negative)
misclassification_rate
Precision = true_positive/(true_positive+false_positive)
Precision
Recall = true_positive/(true_positive+false_negative)
Recall
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
F1_Score

# gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
## create tree object
clf_boost = GradientBoostingClassifier(n_estimators=100,max_depth=10)
## fitting the model
clf_boost = clf_boost.fit(hd_x_train, hd_y_train)
## training dataset is only used for training and test dataset should be used to evaluate the performance of model
predictions = clf_boost.predict(hd_x_test)
print('\nTarget on test data',predictions)
# Use score method to get accuracy of model
## this score is accuracy
score = clf_boost.score(hd_x_test, hd_y_test)
mylist.append(score)
print(score)
from sklearn import metrics
cm = metrics.confusion_matrix(hd_y_test, predictions)
print(cm)
## in column survived, 1 means the person has died and 0 means the person survived

plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)

# most important features in gradient boosting model
Feat_Imp = pd.DataFrame({"Features" : hd_x_train.columns,"Importance" : clf_boost.feature_importances_})
Feat_Imp = Feat_Imp.sort_values(by=['Importance'],ascending=False)
print(Feat_Imp)
#
loca = sns.barplot(x="Importance",y="Features",data = Feat_Imp)
plt.show()
"""
In the above figure we can see that, 14 people actually did not survive and our model also predicted the same
42 people actually survived and our model also predicted the same
9 people did not survive actually, but our model said that they lived
10 people survived actually, but our model said that they died
"""
true_negative = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_positive = cm[1][1]

Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy
misclassification_rate = (false_positive + false_negative) / (true_positive +false_positive + false_negative + true_negative)
misclassification_rate
Precision = true_positive/(true_positive+false_positive)
Precision
Recall = true_positive/(true_positive+false_negative)
Recall
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
F1_Score

# list of accuracy scores for all models
print(mylist)