

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import tree

csv_file_name = "1-15 Processed.csv"

df = pd.read_csv( csv_file_name )
df.head(5)

target_column_name = "HeartDisease"

categorical_target_column = True
#categorical_target_column = False

#Find out number of rows and columns
print(f'Total Number of Rows : {df.shape[0]}')
print(f'Total Number of Columns : {df.shape[1]}')

if categorical_target_column :
  print( df[target_column_name].value_counts() )

"""OBSERVATION

---

There are total 88154 data points in this dataset . Among which 80631 data points belong to class "0" (No Risk of HeartDisease) and other 7523 data points belong to class "1" (HeartDisease).
This indicates that in the dataset the data points are not equally distributed among the classes.
"""

df.isnull().sum().plot(kind='bar')
plt.title("Total Number of Null values for Each Column")

#Find out the percentages of null value cout for each column
#( df.isnull().sum()/len(df) ) *100
print(((df.isnull().sum() / len(df)) * 100).apply(lambda x: f"{x:.2f}%"))

# Assuming 'df' is your DataFrame
df.dropna(how='all', inplace=True)

columns_with_null = ['Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
                     'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race',
                     'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime',
                     'Asthma', 'KidneyDisease', 'SkinCancer']

missing_values_per_column = df[columns_with_null].isnull().sum()
print("Missing values per column:")
print(missing_values_per_column)

rows_with_null = df[df[columns_with_null].isnull().any(axis=1)]
df.drop(rows_with_null.index, inplace=True)

"""OBSERVATION

---


There is no data points or column which has missing values.So, each column can be taken in consideration for further analysis.
"""

# If total number of missing value is less than 5% then drop it otherwise fill using backward fill/forward fill.

print(f'Maximum Null values in column (Before Handling)  : { df.isnull().sum().max() }')

if (df.isnull().sum().max() > len(df) ) * 0.05:
  print("\n------Dropped Null Values-------\n")
  df.dropna( inplace = True)
else:
  print("\n------Replaced Null Values-------\n")
  df.fillna( method = 'bfill' , inplace = True) # You can use 'ffill' to forward fill


print(f'Maximum Null values in column (After Handling)  : { df.isnull().sum().max() }')

#Duplicate entry count
df.duplicated().value_counts()

print( ( df.duplicated().value_counts()/len(df) ) * 100)

"""OBSERVATION

---
There are 1.20% duplicate entries in this dataset.

"""

#Pair Plot Gives you an overall insight on how the data's are distributed
sns.pairplot( df )

"""OBSERVATION

---
From the pair plot above some of the observations are :-   

1.    **Smoking** : From the histogramanalysis it can be observed that the graph is right skewed.Also, mentionably high number of smokers are at class 0.

2.   **AlcoholDrinking ,Asthma and kidneyDisese** : In this dataset, the data points of  AlcoholDrinking are right-skewed where the number of class 0 is very high than class 1.

3.  **PhysicalHealth** : From the histogramanalysis it can be observed that the graph is highly right skewed.Also, mentionably very high number of passengers bordered at class 0.

4.  **Sex** : The observation indicates that the graph is less right skewed. Here the number of male and female which are at class 1 and 0 are near equal same.
4.  **PhysicalActivity** : The observation indicates that the graph is left skewed. Here high number of physical activity is very high in class 1.


"""

df.info()

print("\n\n-----------------Unique Values per column--------------------------------\n\n")

df.nunique()

"""All categorical and numerical colums are given below.Here those coluns have only two types of values 0 or 1, which is binary posibility and that is of course categorical."""

categorical_columns = ["HeartDisease" , "Smoking", "AlcoholDrinking"  ,"DiffWalking" ,"Race","Sex","Diabetic","PhysicalActivity","KidneyDisease","SkinCancer"]
numeric_columns = [ "BMI" , "AgeCategory", "PhysicalHealth" ,"MentalHealth",  "SleepTime","GenHealth"]

if categorical_target_column:
  for column in categorical_columns:
    if column != target_column_name:
      #sns.barplot(x=column, y='Counts', hue= target_column_name, data= df.groupby([column, target_column_name]).size().reset_index(name="Counts"))
      sns.countplot( x = column , hue = target_column_name , data = df )
      plt.show()

  for column in numeric_columns:
    if column != target_column_name:
      sns.histplot( x = column , hue = target_column_name , data = df)
      plt.show()

else:
  for column in categorical_columns:
      if column != target_column_name:
        sns.histplot( x = target_column_name , hue = column , data = df)
        plt.show()

  for column in numeric_columns:
    if column != target_column_name:
      sns.scatterplot( x = target_column_name , y = column , data = df)
      plt.show()

"""OBSERVATION

---


1.   **HeartDisease and smoking**: The count plot indicates that there is no major significant smoker-based difference in the risk of HeartDisease ; both smoker and non smoker exhibit near equal risk.

2.  **HeartDisease and AlcoholDrinking**: The analysis suggests that a higher proportion of individuals with no AlcoholDrinking levels are more prone to heart disease, highlighting a lower risk rate among more AlcoholDrinkers individuals.

3. **HeartDisease and DiffWalking**: The analysis suggests that the HeartDisease risk is high who are not walking ,and the risk is low who are walking.

4.   **HeartDisease and Race**: The count plot suggests that who are not racing their HeartDisease rete is high and the HeartDisease is high who are less racing.

5.  **HeartDisease and Sex** : The count plot indicates that the male HeartDisease rate is higher more than female.

6. **HeartDisease and Diabetc** : Here in this plot we can see that the diabetc patitent is low risk of HeartDisease.

7.   **HeartDisease and physicalActivity**: The count plot suggests that the HeartDisease risk is very high whose physical activity is high.

8.  **HeartDisease and KidneyDisease** : The analysis reveals that whoes have kidneydisease their HeartDisease risk is low.

9. **HeartDisease and SkinCancer** : The analysis reveals that whoes have SkinCancer their HeartDisease risk is low.

10.   **HeartDisease and BMI**: The count plot suggests that individuals with hihg BMI levels  at a higher risk of HeartDisease.

11.  **HeartDisease and AgeCategory** : Analysis reveals that a higher proportion of people at risk of HeartDisease range of 8-10.

12. **HeartDisease and PhysicalHealth** : This analysis indicates that those people physicalHealth condition is not good they have a high risk of HeartDisease.

13.  **HeartDisease and MentalHealth ** : This analysis indicates that those people MentalHealth condition is not good they have a high risk of HeartDisease.

14. **HeartDisease and SleepTime** : Individuals with a sleeping range of 6-8 are at a higher risk of HeartDisease.

15.   **HeartDisease and GenHealth**: This analysis indicates that the range of 3 and 4 is high risk of HeartDisease.


"""

if categorical_target_column:
  sns.pairplot( data=df , hue = target_column_name)

"""OBSERVATION

---

**AlcoholDrinking and HeartDisese** : Individuals who are not taking blood pressure medications (BPMeds) are at a higher risk of TenYearCHD.

**AgeCategory and HeartDisease** : The graph indicates that the age category between 5-15 are highly risk in HeartDisease.

**SleepTime and HeartDisease** : The graph indicates that the SleepTime range between 5-10 are highly risk in HeartDisease.

**Sex and HeartDisese** : Here in this graph we can see that the female has low risk than the male.

**BMI and HeartDisease** : The graph indicates that the NBI range between 15-30 are highly risk in HeartDisease.

"""

df.info()

#Correlation HeatMap for numeric columns among the dataset
sns.heatmap(df.corr( numeric_only =  True))

"""OBSERVATION

---
Based on the correlation matrix, it appears that certain variables like GenHealth, DiffWalking,PhysicalHealth,BMI,Race,Diabetic,KidneyDisease and skincancer have a more pronounced correlation with HeartDisease.

Features and labels are stored in different variables. Categorical columns are encoded using OrdinalEncoder .
"""

X = df.drop(target_column_name , axis=1 )
y =  df[target_column_name]

enc = OrdinalEncoder()
X = enc.fit_transform( X )


le = LabelEncoder()
target_class = y.unique()
y = le.fit_transform( y )

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)

from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt

if categorical_target_column:
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, zero_division=1))

    matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(matrix)
    plt.show()

    sns.barplot(x=target_class, y=recall_score(y_test, y_pred, average=None, zero_division=1))
    plt.title("Class Wise Recall Score (SVM)")
    plt.show()

    sns.barplot(x=target_class, y=precision_score(y_test, y_pred, average=None, zero_division=1))
    plt.title("Class Wise Precision Score (SVM)")
    plt.show()

else:
    # Apply linear regression. As shown in the lab
    print("You need to use Linear Regression as your target column is numeric.")

"""OBSERVATION

---
"The evaluation metrics demonstrate an exceptional performance for class '0,' with high precision, recall, and F1-score, signifying the model's excellent ability to accurately identify instances belonging to this class. However, concerning class '1,' while it exhibits a precision of 1.00, the recall and F1-score are notably low at 0.00. This outcome suggests that although the model identifies '1' instances with perfect precision, it fails entirely to recall any positive instances from this class. This discrepancy underlines a critical issue where the model excels in correctly classifying '0' instances but completely overlooks or misclassifies all '1' instances, indicating a substantial limitation in recognizing this specific class."


"""

if categorical_target_column:
  clf = tree.DecisionTreeClassifier()
  clf.fit( X_train , y_train )
  y_pred = clf.predict( X_test )

  print( classification_report( y_test , y_pred ) )

  matrix = confusion_matrix( y_test , y_pred )
  sns.heatmap( matrix )
  plt.show()

  sns.barplot( x = target_class ,y = recall_score( y_test , y_pred , average =  None) )
  plt.title( "Class Wise Recall Score (Decision Tree)")
  plt.show()

  sns.barplot( x = target_class ,y = precision_score( y_test , y_pred , average =  None) )
  plt.title( "Class Wise Precision Score (Decision Tree)")
  plt.show()

else:
  #apply linear regression . As shown in lab
  print("You Need to use Linear Regression as your target column in Numeric")

"""OBSERVATION

---

The decision tree model applied to the heart disease dataset achieved an accuracy of 86%.
 The precision is higher for class 0 at 93% than for class 1 at 22%.The recall is aslo higher for class 0 at 92% than for class 1 at 25%. The F1-score, considering both precision and recall, was 92% for class 0 and 24% for class 1.
"""