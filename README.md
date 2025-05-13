
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas

2. Import Decision tree classifier

3. Fit the data in the model

4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: M ALMAAS JAHAAN
RegisterNumber:  212224230016
*/
import pandas as pd
df=pd.read_csv("/content/Employee.csv")
print("data.head():")
df.head()
print("data.info()")
df.info()
print("data.isnull().sum()")
df.isnull().sum()
print("data value counts")
df["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
df["salary"]=le.fit_transform(df["salary"])
df.head()
print("x.head():")
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=df["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
print("Data prediction")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt,filled=True,feature_names=x.columns,class_names=['salary' , 'left'])
plt.show()

```

## Output:
![decision tree classifier model](sam.png)

![443105621-3d26bd40-3cd8-4b32-80ad-c00c8c95bd5f](https://github.com/user-attachments/assets/060770e6-a7b2-4882-b3c5-3a21ad470794)

![443105839-7d66ff5e-687f-439d-93be-426bcb395869](https://github.com/user-attachments/assets/40c752f1-2585-4dac-a380-26a849f5c81a)

![443106081-bf1b468a-d587-483e-bd10-17f7d0429299](https://github.com/user-attachments/assets/766e65de-5ffb-48d3-a265-66813065c41b)

![443106549-a1b5d786-0b63-4146-913a-57cea53b75dc](https://github.com/user-attachments/assets/6aab962f-0e33-467f-9821-3eefce7dba9b)

![443106813-229f6cd8-8558-4b47-abfa-c720f22e5eb6](https://github.com/user-attachments/assets/127abaa5-71fe-4d93-ae39-044eba4fba2a)

![443107113-e3fc21b7-58d4-4ac9-9043-bc667972bb65](https://github.com/user-attachments/assets/9fcb71aa-f6dc-48f9-a7f8-adc2e0594814)

![443107431-80148c1f-92a8-4f79-85c2-6e325b288614](https://github.com/user-attachments/assets/8f3aed06-248e-42ef-8b3d-f959cfbb8550)

![443107677-813bb322-c02d-4a39-90bb-105d6b67ec65](https://github.com/user-attachments/assets/ff5d03ad-178a-4e40-908c-eb813f5eef74)

![443107780-95f47f2d-dba2-4527-b719-d01aea66d3f1](https://github.com/user-attachments/assets/5c90b4df-eb94-4442-b4c0-430cc6e1644c)

![443108030-067495d8-4ff5-408c-8675-79e2f2b50f15](https://github.com/user-attachments/assets/68809148-8383-4a50-a9e7-597d785e42b3)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
