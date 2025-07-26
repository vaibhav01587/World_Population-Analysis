import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
# LOAD DATA SET
df=pd.read_csv("world_population.csv",encoding="LATIN1")
print(df)
#DISPLAY BASIC INFORMATION ABOUT THE DATASET
print(df.info())
print(df.head(10))

#DATA PREPROCESSING 
# HANDLING MISSING VALUES
print(df.isnull().sum())
# there is no missing values is zer

# duplicated value is also zero
print(df.duplicated().sum())

print(df.columns)

#to remove column capital  nd cca3 from dataset
df.drop(['Capital','CCA3'],axis=1,inplace=True)
print(df.columns)

#To view top 5 columndata
print(df.head())

#To view bottom 5 data
print(df.tail())

custom_palette = ['#0b3d91', '#e0f7fa', '#228b22', '#1e90ff', '#8B4513', '#D2691E',
'#DAA520', '#556B2F']


Countries_by_continent=df['Continent'].value_counts().reset_index()
print(Countries_by_continent)


# converting area to type float

plt.figure(figsize=(8, 6))
sns.barplot(x='Continent', y='2022 Population', data=df, estimator=sum)
plt.title('Total Population by Continent in 2022')
plt.xticks(rotation=45)
plt.show()

# converting area to type float


def convert_area(area):
    if isinstance(area, str):
        if area == '< 1':
            return 0.5
        if 'M' in area:
            return float(area.replace('M', '')) * 1000000
        elif 'K' in area:
            return float(area.replace('K', '')) * 1000
    return float(area)
df['Area (km2)'] = df['Area (km2)'].apply(convert_area)
df["pop change"] = df["2022 Population"]- df["2020 Population"]
x = df[['2020 Population', 'Area (km2)', 'Density (/km2)', 'Growth Rate', 'World Population Percentage']]
y = df['2022 Population']

#Plotting Population 2024 vs count
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
sns.histplot(df['2020 Population'], kde=True, ax=axs[0, 0])
axs[0, 0].set_title('Distribution of Population 2020')

#Plotting Area vs count
sns.histplot(df['Area (km2)'], kde=True, ax=axs[0, 1])
axs[0, 1].set_title('Distribution of Area (km²)')

#Plotting Density 2024 vs count
sns.histplot(df['Density (/km2)'], kde=True, ax=axs[1, 0])
axs[1, 0].set_title('Distribution of Density (/km²)')

#Plotting Growth rate vs count
sns.histplot(df['Growth Rate'], kde=True, ax=axs[1, 0])
axs[1, 1].set_title('Distribution of Growth Rate')

plt.tight_layout()
plt.show()

sns.barplot(x='Country/Territory', y='2022 Population',data=df)

#correlation matrix using numeric columns

numeric_cols = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_cols.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# Pre-Processing  Model Training


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x_train)
x_scaled_test=scaler.transform(x_test)
print("Model training using linear regression")

regressor=LinearRegression()
reg=regressor.fit(x_scaled,y_train)
print("REG SCORE")
print(reg.score(x_scaled_test,y_test))

y_pred1=reg.predict(x_scaled_test)
print("MEAN SQUARED ERROR")
print(mean_squared_error(y_test,y_pred1))
print("R2 SCORE")
print(r2_score(y_test,y_pred1))


from sklearn.tree import DecisionTreeRegressor
print("Model training using DECISION TREE REGRESSOR")
clf = DecisionTreeRegressor()
clf.fit(x_scaled,y_train)

print("Model Training using decision tree")
print(clf.score(x_test,y_test))

y_pred2=clf.predict(x_test)
print("MEAN SQUARED ERROR")
print(mean_squared_error(y_test,y_pred2))

print("R2 SCORE")
print(r2_score(y_test,y_pred2))

from sklearn.ensemble import RandomForestRegressor

print("Model training using RANDOM FOREST REGRESSOR")
clf=RandomForestRegressor()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))

y_pred3=clf.predict(x_test)

print("MEAN SQUARED ERROR")
print(mean_squared_error(y_test,y_pred3))
print("R2 SCORE")
print(r2_score(y_test,y_pred3))



from sklearn.svm import SVR
print("Model training using SVR")
clf=SVR()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
y_pred4=clf.predict(x_test)
print("MEAN SQUARED ERROR")
print(mean_squared_error(y_test,y_pred4))
print("R2 SCORE")
print(r2_score(y_test,y_pred4))

# Linear regression has least mean squared error

print(type(y_pred1))
print(type(y_test))
print(type(x_test))
print(type(x_scaled))

y_pred1=pd.Series(y_pred1,index=y_test.index,name='y_pred')
df_new=pd.concat([x_test,y_test.rename('y_test'),y_pred1],axis=1)
df_new.reset_index(drop=True,inplace=True)
df_new.head()

df_new.to_csv('World Population 2022 REPORT.csv',header=True,index=True)
