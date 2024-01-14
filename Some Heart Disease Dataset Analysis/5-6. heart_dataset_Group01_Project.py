

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

csv_file_name = "5-6. heart.csv"

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

There are total 918 data points in this dataset . Among which 508 data points belong to class "0" (HeartDiseases - Absent) and other 410 data points belong to class "1" (HeartDiseases - Present).
This indicates that in this data is balanced for both heart disease presence and absence.
"""

df.isnull().sum().plot(kind='bar')
plt.title("Total Number of Null values for Each Column")

#Find out the percentages of null value cout for each column
( df.isnull().sum()/len(df) ) *100

"""OBSERVATION

---

This dataset doesn't have any missing values. So no need any missing value handling techniques.
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

** Age: Age and resting BP graph indicating that as age increases, resting blood pressure tends to be higher.

** Cholesterol: This variable is not significantly various with Heart Disease in this dataset.

** MaxHR: It shows that younger people are have higher maximum heart rate.

** OldPeak: Which is numeric representation of depression level, and it shows people with more depression are more likely to be have heart failure.


"""

df.info()

print("\n\n-----------------Unique Values per column--------------------------------\n\n")

df.nunique()

categorical_columns = ["Sex" , "ChestPainType" , "FastingBS" ,"RestingECG" ,"ExerciseAngina" , "ST_Slope", "HeartDisease"]
numeric_columns = [ "Age" , "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

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


1.   **HeartDisease and Sex**: The count plot reveals an imbalance in gender distribution, with a higher number of males than females in the dataset. Consequently, males show a higher prevalence of heart disease.

2.  **HeartDisease and ChestPainType** : Among the four types of chest pain, individuals experiencing Asymptomatic chest pain (Type 4) show a significantly higher incidence of heart disease.

3. **FastingBS and HeartDisease** : The variable "FastingBS over 120" appears imbalanced, with a disproportionately higher number of data points in the 0 class, indicating lower fasting blood sugar levels.

4.   **HeartDisease and ExcerciseAngina**: The count plot suggests that individuals engaging in exercise may be more susceptible to heart disease.

5.  **HeartDisease and ST_Slpo** : The dataset includes three types of ST_Slpo, with individuals exhibiting a normal ST segment slope facing a higher risk of heart disease.

6. **Age and HeartDisease** : People within the age group of 45-65 exhibit the highest risk of developing heart disease.

7.   **HeartDisease and RestingBP**: Individuals with a resting blood pressure in the range of 120-150 show an elevated likelihood of heart disease.  

8.  **HeartDisease and Cholesterol** : Those with cholesterol levels ranging from 200-300 face a higher risk of heart disease.

9. **FastingBS and MaxHR** : Individuals with a fluctuating maximum heart rate between 120-160 are at a considerably higher risk of heart failure.

"""

if categorical_target_column:
  sns.pairplot( data=df , hue = target_column_name)

"""OBSERVATION

---

** Among the four types of chest pain, individuals experiencing Asymptomatic chest pain (Type 4) show a significantly higher incidence of heart failure.

** Middle age group people are having more heart failure.

** whenever resting blood pressure cross 130 its high risk of having heart failure.

** Older people with more cholesterol are at top most risk of having heart failure.

"""

df.info()

#Correlation HeatMap for numeric columns among the dataset
sns.heatmap(df.corr( numeric_only =  True))

"""OBSERVATION

---

The correlation analysis of this heart disease prediction dataset reveals various relationships between different health parameters and the likelihood of heart disease. A strong positive correlation is observed between Age and RestingBP, indicating that as age increases, resting blood pressure tends to be higher. Cholesterol shows a moderate positive correlation with Age and RestingBP but is not significantly correlated with Heart Disease in this dataset. MaxHR (Maximum Heart Rate) exhibits a negative correlation with Age, suggesting that younger individuals tend to have higher maximum heart rates. Oldpeak has a noticeable positive correlation with Heart Disease, indicating that individuals with higher Oldpeak values are more likely to have heart disease.

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

An analysis of the evaluation metrics reveals a notable discrepancy in recall scores between the two classes. Class "0" demonstrates a recall of 0.70, while class "1" exhibits a slightly higher recall of 0.79. This suggests that the model excels in accurately identifying instances within class "1" but faces challenges in capturing instances of class "0" with the same level of proficiency.

The precision scores reflect this pattern, with class "1" having a higher precision of 0.79 compared to class "0" at 0.69. This indicates that when the model predicts instances as belonging to class "1," it is correct more often than when predicting class "0."

The overall accuracy of the model is 75%, showcasing its ability to make correct predictions across both classes. The macro-average precision, recall, and F1-score are consistent at 0.74, indicating a balanced performance. The weighted-average metrics, considering the class distribution, are also at 0.75.
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


The decision tree model applied to the heart disease detection dataset achieved an accuracy of 74%. Notably, it demonstrated higher precision for instances without heart disease (class 0) at 65% compared to those with heart disease (class 1) at 84%. The recall was higher for class 0 at 80% than for class 1 at 70%. The F1-score, considering both precision and recall, was 0.72 for class 0 and 0.76 for class 1.

This analysis suggests that the model performs reasonably well in identifying instances without heart disease, with a balanced precision and recall. However, there is room for improvement in predicting instances with heart disease, as reflected by the higher precision but slightly lower recall for class 1. Further exploration and potential model refinement may enhance the predictive capabilities, especially in capturing cases of heart disease.
"""