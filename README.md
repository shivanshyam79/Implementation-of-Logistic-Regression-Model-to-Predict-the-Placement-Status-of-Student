# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## step 1. Import the required packages and print the present data.
## step 2. Print the placement data and salary data.
## step 3. Find the null and duplicate values.
## step 4. Using logistic regression find the predicted values of accuracy , confusion matrices.
## step 5. Display the results. 
## step 6 stop
## Program:
```
/*
## Developed by: shyam R
## RegisterNumber: 212223040200

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


*/
```

## Output:
## Head
![325658586-eefa62b7-f638-4ca7-a02f-46fc2e53cdec](https://github.com/shivanshyam79/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151513860/ac5bdbf1-fed2-4b43-a437-51d100cade27)
## After removing sl_no , salary
![325658745-be21cb66-40f2-485d-936a-5105265df776](https://github.com/shivanshyam79/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151513860/3d2c7d0d-8985-42fa-a0c5-6fdc061237b0)
## Null data
![325658891-33431ddf-c114-4f35-a73f-34aef59c2abf](https://github.com/shivanshyam79/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151513860/de2b0413-94c9-47da-b591-768eec278cc7)
## Duplicated sum
![325659076-4edbb4d0-ec2e-4e45-a62b-4e32e72a5c5e](https://github.com/shivanshyam79/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151513860/50853e29-5e76-425d-adf6-8ec8d6b08314)
## Label Encoder
![325659242-c813c224-8210-4868-bb71-21b3dd89a987](https://github.com/shivanshyam79/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151513860/17508ff6-aa06-4e85-97b9-32eb02537ef1)
## After removing the last column
![325659461-ccc79230-9ad7-4554-9b58-7e233550d787](https://github.com/shivanshyam79/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151513860/d6362dd7-d7aa-437d-b946-cb2deb321ccf)
## Displaying the status
![325659688-8c9554ab-a951-4f27-88b1-f133bb828fb1](https://github.com/shivanshyam79/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151513860/cb535c8f-e288-4c06-bb69-4183f3a50a12)
## Prediction of y
![325659838-b1681be5-0496-4a4d-8f23-1cdc35f73079](https://github.com/shivanshyam79/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151513860/f1f02912-78ae-4cf7-9021-0ab2d3faa51b)
## Accuracy score
![325659963-30d6b905-4409-433d-8b79-edb7b5215df0](https://github.com/shivanshyam79/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151513860/c968e825-5710-4ff0-a09f-d400392f1194)
## Confusion
![325660387-56d033b3-e691-492e-a2e8-ea641518f60b](https://github.com/shivanshyam79/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151513860/98ce5c7b-949f-4d93-8708-7c9e791682f2)
## Classification report
![325660784-a710f9c6-f40b-4588-9971-ba75d3ed0704](https://github.com/shivanshyam79/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151513860/4d1b7fe1-d3c3-42be-9a06-f06bd4886619)
## Prediction
![325660936-42e740b2-8e40-489a-b04c-50e4b76ce393](https://github.com/shivanshyam79/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151513860/19c88f8f-1a8d-4141-8110-9552cb87edfd)
## Result
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
