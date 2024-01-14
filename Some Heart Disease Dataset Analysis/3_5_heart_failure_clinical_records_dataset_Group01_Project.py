

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

csv_file_name = "3-5. heart_failure_clinical_records_dataset.csv"

df = pd.read_csv( csv_file_name )
df.head(5)

target_column_name = "DEATH_EVENT"

categorical_target_column = True
#categorical_target_column = False

#Find out number of rows and columns
print(f'Total Number of Rows : {df.shape[0]}')
print(f'Total Number of Columns : {df.shape[1]}')

if categorical_target_column :
  print( df[target_column_name].value_counts() )

"""OBSERVATION

---

There are total 299 data points in this dataset . Among which 203 data points belong to class "0" (not Dead) and other 96 data points belong to class "1" (Dead).
This indicates that in the dataset the data points are not equally distributed among the classes.
"""

df.isnull().sum().plot(kind='bar')
plt.title("Total Number of Null values for Each Column")

#Find out the percentages of null value cout for each column
( df.isnull().sum()/len(df) ) *100

"""OBSERVATION

---

This dataset doesn't have any missing values in any of the columns. So, we don't need to drop any columns or don't need to use any technique for handling missing values as it doesn't has any missing values.
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

1.    **Platelates** : From the histogram analysis it can be observed that the graph is symetric.

2.   **Ejection Fraction** : In this dataset the data points from `Fare` column are left skewed.

3.  **Smoking** : This graph show a right skewed,

4.   **Serum Creatinine** :This is also shows a right skewed graph

5.  **High Blood Pressure** : This graph show a right skewed.


"""

df.info()

print("\n\n-----------------Unique Values per column--------------------------------\n\n")

df.nunique()

""" **[FILLOUT] From the information above fill out the categorical column and numeric column names.**  <br>

Question: How do I know if the column have categorical data or not? <br>
Ans:  If the column datatype is "object" it's more likely to be categorical. Any data type other than "object might not be categorical column. <br>
In this dataset , the "Survived" column has datatype "int64". It might seem that the column has numeric data type . But if we oberserve carefully it is obvivous that the "Survived" column is categorical.  

Also, if the column has Very Few Unique values then also it's more likely to be categorical. Last, you use your intuition.
  
*DELETE THIS TEXT SECTION BEFORE SUBMITTING*
"""

categorical_columns = ["anaemia" , "diabetes" , "high_blood_pressure" ,"sex" ,"smoking" , "DEATH_EVENT"]
numeric_columns = [ "age" , "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time"]

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


1.   **DEATH_EVENT and anemia**: From the count plot it can be observed that it has not much of variations, both having and not having has the same number of death result.

2.  **DEATH_EVENT and diabates** : Analysis reveals that a higher proportion of people without diabates has more number of death.

3. **HighBP and DEATH_EVENT** : It shows high blood pressure somehow is not any cause of heart disease death as not having high blood pressure people are higher with heart disease failure death.

4.   **DEATH_EVENT and sex**: Graph shows us that male are having more death with heart failure than female.

5. **Age and DEATH_EVENT** : From the plot we find that age group 60-70 having more death as a result of heart failure.

7.   **DEATH_EVENT and creatinine_phosphokinase**: Creatinine_phosphokinase value with less than 700 are having higher number of death.

8.  **DEATH_EVENT and ejection_fraction** : Analysis reveals that ejection fraction value range 20-40 are having higher number of death as a result of heart failure.

9. **DEATH_EVENT and platelets** : 2 to 3 lacs platelets range people are dying more from heart failure.

10.  **DEATH_EVENT and serum creatinen** : Analysis reveals that whoever having serum creatinen level 1-1.8 are having death highly from an heart failure.

11. **DEATH_EVENT and serum sodium** : Serum sodium level 135-140 are shown the most dying cause from an heart failure.
"""

if categorical_target_column:
  sns.pairplot( data=df , hue = target_column_name)

"""OBSERVATION

---


2.  **diabates** : A lot more data with no diabates are here than having diabates.

3. **HighBP** : Dataset is also imbalanced for HighBP with more people with having high BP present than not having High BP

4.   **sex**: Male are having more death.

5. **Age and DEATH_EVENT** : Older people dying number is high.

9. **Serum creatinine and age** : Older people with high serum creatinine are dying more.
"""

df.info()

#Correlation HeatMap for numeric columns among the dataset
sns.heatmap(df.corr( numeric_only =  True))

"""OBSERVATION

---
Based on the correlation matrix, it appears that serum_creatinine and age have a relatively higher positive correlation with DEATH_EVENT, indicating that as these variables increase, the likelihood of death (DEATH_EVENT=1) also increases. Time has a negative correlation with DEATH_EVENT, suggesting that as time increases, the likelihood of death decreases.

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
The evaluation metrics highlight a significant disparity in recall between the two classes. Class "0" demonstrates strong recall at 0.92, showcasing its ability to effectively identify instances. In contrast, class "1" exhibits a lower recall of 0.49, indicating challenges in capturing instances within this category. While the precision for class "1" is higher at 0.82, suggesting correctness in positive predictions, the lower recall implies potential underrepresentation of actual instances.

The overall accuracy of the model is 74%, reflecting a balanced performance across both classes. However, there is evident room for improvement, particularly in enhancing the model's capability to correctly identify instances belonging to class "1." Strategic adjustments and refinements could further optimize the model's predictive capacity and address the observed recall discrepancy.
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

The decision tree model applied to the heart failure clinical records dataset achieved an accuracy of 69%. Notably, it demonstrated higher precision for patients who did not experience death (class 0) at 70%, compared to those who did (class 1) at 66%. The recall was higher for class 0 at 81% than for class 1 at 51%. The F1-score, considering both precision and recall, was 0.75 for class 0 and 0.58 for class 1.

This analysis suggests that the model performs reasonably well in identifying instances where patients do not experience death, with higher precision and recall for class 0. However, there is room for improvement in predicting instances of death (class 1), as reflected by lower precision and recall. Further exploration and potential model refinement may enhance the predictive capabilities, especially in identifying cases of death in heart failure clinical records.
"""