
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#load data
df = pd.read_csv("Walmart_sales.csv")
df

#Count Plot of Holiday vs. Non-Holiday Weeks:

# datetime format
df['Date'] = pd.to_datetime(df['Date'])

# new column 'Holiday' based on 'Holiday_Flag' values
df['Holiday'] = df['Holiday_Flag'].apply(lambda x: 'Holiday' if x == 1 else 'Non-Holiday')

# count plot
plt.figure(figsize=(8, 6))
sns.countplot(x='Holiday', data=df, color='b')

# labels and title
plt.title('Count Plot of Holiday vs. Non-Holiday Weeks')
plt.xlabel('Week Type')
plt.ylabel('Count')
plt.show()

#Bar Plot of Total Sales by Store:

# total sales by store
total_sales_by_store = df.groupby('Store')['Weekly_Sales'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(x='Store', y='Weekly_Sales', data=total_sales_by_store, color='b')

plt.title('Bar Plot of Total Sales by Store')
plt.xlabel('Store')
plt.ylabel('Total Sales')
plt.show()

#Pie Chart of Holiday vs. Non-Holiday Sales Proportion:

total_holiday_sales = df.loc[df['Holiday_Flag'] == 1, 'Weekly_Sales'].sum()
total_non_holiday_sales = df.loc[df['Holiday_Flag'] == 0, 'Weekly_Sales'].sum()

labels = ['Holiday Sales', 'Non-Holiday Sales']
sizes = [total_holiday_sales, total_non_holiday_sales]
colors = ['#ff9999','#66b3ff']

plt.figure(figsize=(7, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)

plt.title('Proportion of Holiday vs. Non-Holiday Sales')
plt.show()

#Count Plot of Sales by Month:

df['Date'] = pd.to_datetime(df['Date'])

df['Month'] = df['Date'].dt.month_name()

plt.figure(figsize=(12, 6))
sns.countplot(x='Month', data=df, color='b', order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])

plt.title('Count Plot of Sales by Month')
plt.xlabel('Month')
plt.ylabel('Count')

plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()

#Bar Plot of Total Sales by Month:

df['Date'] = pd.to_datetime(df['Date'])

df['Month'] = df['Date'].dt.month_name()

total_sales_by_month = df.groupby('Month')['Weekly_Sales'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(x='Month', y='Weekly_Sales', data=total_sales_by_month, color='b', order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])

plt.title('Total Sales by Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')

plt.xticks(rotation=45)
plt.show()

#Pie Chart of Sales Proportion by Month:

df['Date'] = pd.to_datetime(df['Date'])

df['Month'] = df['Date'].dt.month_name()

total_sales_by_month = df.groupby('Month')['Weekly_Sales'].sum()

plt.figure(figsize=(8, 8))
plt.pie(total_sales_by_month, labels=total_sales_by_month.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)

plt.title('Sales Proportion by Month')
plt.show()

#Count Plot of Sales by Day of the Week:

df['Date'] = pd.to_datetime(df['Date'])

df['Day_of_Week'] = df['Date'].dt.day_name()

plt.figure(figsize=(10, 6))
sns.countplot(x='Day_of_Week', data=df, color='b', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

plt.title('Count Plot of Sales by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Count')

plt.xticks(rotation=45)
plt.show()

#Bar Plot of Total Sales by Day of the Week:

df['Date'] = pd.to_datetime(df['Date'])

df['Day_of_Week'] = df['Date'].dt.day_name()

total_sales_by_day = df.groupby('Day_of_Week')['Weekly_Sales'].sum().reset_index()

days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

plt.figure(figsize=(10, 6))
sns.barplot(x='Day_of_Week', y='Weekly_Sales', data=total_sales_by_day, order=days_order, color='b')

plt.title('Total Sales by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Total Sales')

plt.xticks(rotation=45)
plt.show()

#Pie Chart of Sales Proportion by Day of the Week:

df['Date'] = pd.to_datetime(df['Date'])

df['Day_of_Week'] = df['Date'].dt.day_name()

total_sales_by_day = df.groupby('Day_of_Week')['Weekly_Sales'].sum()

plt.figure(figsize=(8, 8))
plt.pie(total_sales_by_day, labels=total_sales_by_day.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Accent.colors)

plt.title('Sales Proportion by Day of the Week')
plt.show()

#5 bonus

#1. Time Series Plot of Weekly Sales:

plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['Weekly_Sales'], marker='o', linestyle='-', color='b')
plt.title('Time Series Plot of Weekly Sales')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.show()

#2. Box Plot of Weekly Sales by Store:

plt.figure(figsize=(14, 8))
sns.boxplot(x='Store', y='Weekly_Sales', data=df, palette='Set2')
plt.title('Box Plot of Weekly Sales by Store')
plt.xlabel('Store')
plt.ylabel('Weekly Sales')
plt.show()

#3. Pair Plot of Numeric Variables:

sns.pairplot(df[['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']], height=2.5)
plt.suptitle('Pair Plot of Numeric Variables', y=1.02)
plt.show()

#4. Correlation Heatmap:

plt.figure(figsize=(10, 8))
correlation_matrix = df[['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

#5. Bar Plot of Average Weekly Sales by Holiday:

avg_sales_by_holiday = df.groupby('Holiday_Flag')['Weekly_Sales'].mean().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(x='Holiday_Flag', y='Weekly_Sales', data=avg_sales_by_holiday, color='b')
plt.title('Average Weekly Sales by Holiday')
plt.xlabel('Holiday')
plt.ylabel('Average Weekly Sales')
plt.xticks(ticks=[0, 1], labels=['Non-Holiday', 'Holiday'])
plt.show()