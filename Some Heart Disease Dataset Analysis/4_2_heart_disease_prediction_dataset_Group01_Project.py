

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

csv_file_name = "4-2. Heart_Disease_Prediction.csv"

df = pd.read_csv( csv_file_name )
df.head(5)

for column in df.columns:
  if( df[column].dtype == "object"):
    grp1 = LabelEncoder()
    grp1.fit( df[column] )
    df[column] = grp1.transform( df[column])
df

target_column_name = "Heart Disease"

categorical_target_column = True
#categorical_target_column = False

#Find out number of rows and columns
print(f'Total Number of Rows : {df.shape[0]}')
print(f'Total Number of Columns : {df.shape[1]}')

if categorical_target_column :
  print( df[target_column_name].value_counts() )

"""OBSERVATION

---

There are total 270 data points in this dataset . Among which 150 data points belong to class "0" (Don't have Heart Disease) and other 120 data points belong to class "1" (Heart Disease) .
This indicates that in the dataset the data points are equally distributed among the classes.


"""

df.isnull().sum().plot(kind='bar')
plt.title("Total Number of Null values for Each Column")

#Find out the percentages of null value cout for each column
( df.isnull().sum()/len(df) ) *100

"""OBSERVATION

---


There are  no missing value in this dataset.So we don't need to drop or using different teachniques for any column.


"""

#Duplicate entry count
df.duplicated().value_counts()

print( ( df.duplicated().value_counts()/len(df) ) * 100)

"""OBSERVATION

---
There are no duplicate entries in this data set

"""

#Pair Plot Gives you an overall insight on how the data's are distributed
sns.pairplot( df )

"""OBSERVATION

---
From the pair plot above some of the observations are :-   

1.    **ST Depression** : From the histogram analysis it can be observed that the graph is right skewed. It also shows that younger people are having more ST Depression than older people.

2.   **Chest Pain Type** : In this dataset the data points from `Fare` column are highly right skewed.

3.  **Age** : Age has a moderatly symetric graph. It also shows Older people hare facing more with Maximum Heart Rate.

4.  **Number of Vessel fluro** : Graph shows a highly right skewed for this variable.

5.  **Cholesterol** : Cholesterol has a symetric graph.

"""

df.info()

print("\n\n-----------------Unique Values per column--------------------------------\n\n")

df.nunique()

categorical_columns = ["Sex" , "Chest pain type" , "FBS over 120" ,"EKG results" ,"Exercise angina" , "Slope of ST" ,"Number of vessels fluro" ,"Thallium" ,"Heart Disease"]
numeric_columns = [ "Age" , "BP" ,"Cholesterol" ,"Max HR" ,"ST depression"]

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


1.**Heart disease and sex:** From the count plot it can be observed the number of male is more than a female who has heart diseases. Here 1 is for male and 0 is for female.

2.**Heart disease and chest pain types:** There are four
values are available .value 1 typical angina ,valid 2 atypical angina ,value 3 non anginal pain and value 4 asymptomatic. Here from the chart we find that asymptomatic chest pain paients are more having heart disease

3.**Heart disease and FBS over 120:** From the chat we can saw that the number of  who has FBS over 120 they have more heart diseases.

4.**Heart diseases and e EKG results: **From the graph we saw that who have more yearly ECG reading they have higher number of heart diseases.

5.**Heart diseases and exercise angina:** From the count plot it can observe this pair is not much of variation.

6.**Heart disease and Slope of ST:**Here value 2 which is flat peak exercise ST segment, and this has more heart diseases number than other two segment.

7.**Heart disease and Thallium:**From the graph we saw that who's thalassemia is around 7 they are reversable defect and they have more heart problems.

8.**Heart disease and age:** Here we saw that the people age is around 50 to 60 they having more heart problems.

9.**Heart disease and BP:** people who having BP over then 120 they are having more heart diseases.

10.**Heart disease and cholesterol:** Cholesterol rate of 200 to 300 people have heart problems more .

11.**Heart disease and Max HR:**The range of Maximum heart rate between 150-170 are having higher level of heart disease.

12.**Heart disease and ST depression:** People having ST depression level from 0.1 to 1.8 having higher numbers of heart disease.
"""

if categorical_target_column:
  sns.pairplot( data=df , hue = target_column_name)

"""OBSERVATION

---


1. **sex:** Male data is higher then female, imbalance shows here.

2. **FBS over 120:** From the chat we can saw that this variable also have imbalance, as lot more data with not having FBS over 120.

3. **ST depression: ** When ST depression level increase asymptomatic chest pain shown more.
"""

df.info()

#Correlation HeatMap for numeric columns among the dataset
sns.heatmap(df.corr( numeric_only =  True))

"""OBSERVATION

---

Based on the correlation matrix it appears that certain variables like age, sex, chest pain type, and exercise angina have noticeable correlations with heart disease. For instance, there is a noticeable positive correlation between age and blood pressure (BP), indicating that as age increases, BP tends to rise as well. Similarly, exercise angina shows a strong positive correlation with ST depression and the number of vessels detected through fluoroscopy. On the contrary, Max HR (Maximum Heart Rate) exhibits a negative correlation with age, implying that younger individuals tend to have higher maximum heart rates.

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

if categorical_target_column:
  clf = SVC()
  clf.fit( X_train , y_train )
  y_pred = clf.predict( X_test )

  print( classification_report( y_test , y_pred ) )

  matrix = confusion_matrix( y_test , y_pred )
  sns.heatmap( matrix )
  plt.show()

  sns.barplot( x = target_class ,y = recall_score( y_test , y_pred , average =  None) )
  plt.title( "Class Wise Recall Score (SVM)")
  plt.show()

  sns.barplot( x = target_class ,y = precision_score( y_test , y_pred , average =  None) )
  plt.title( "Class Wise Precision Score (SVM)")
  plt.show()

else:
  #apply linear regression . As shown in lab
  print("You Need to use Linear Regression as your target column in Numeric")

"""OBSERVATION

---

An analysis of the evaluation metrics reveals a significant discrepancy in recall scores between the two classes. Class "0" has a precision of 0.74, signifying a 74% accuracy in its predictions for instances classified as "0." Then, the recall for class "0" stands at 0.82, meaning that 82% of all actual instances of class "0" are correctly identified by the model. The harmonic mean of precision and recall, denoted as the F1-score, consolidates at 0.78 for class "0," reflecting a balanced performance between precision and recall.

Then, class "1" presents a slightly lower precision of 0.67, indicating a 67% accuracy in predictions for instances classified as "1." The recall for class "1" settles at 0.56, indicating that roughly 56% of actual instances of class "1" are accurately captured by the model. The F1-score for class "1" culminates at 0.61, reflecting a moderate balance between precision and recall for this class.

Finally, the model achieves an overall accuracy of 0.72, signifying the correctness of predictions across both classes. The macro-average, considering an equal weightage for both classes, yields an F1-score of 0.69. Meanwhile, the weighted average, considering the influence of class imbalance due to varying support, results in an F1-score of 0.71. These collective metrics emphasize a moderately balanced performance of the model, albeit with a tendency towards better prediction for class "0" instances.


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
With an overall accuracy of 72%, the model exhibits stronger precision (73%) and recall (84%) for class "0" compared to class "1," where precision stands at 68% and recall at 53%. These differences in performance reflect in the F1-scores, culminating at 0.78 for class "0" and 0.60 for class "1". The model notably excels in accurately identifying instances of class "0" but displays room for improvement in effectively capturing instances of class "1". This suggests a potential area for refining the model's performance, particularly in enhancing its ability to recognize instances categorized as class "1" more accurately.
"""