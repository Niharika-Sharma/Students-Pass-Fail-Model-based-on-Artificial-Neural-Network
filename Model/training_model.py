###############################################################################################################
#### Data Refining and buildin an ANN (Artificial Neural Network) Model
###############################################################################################################
import pandas as pd
import numpy as np

df = pd.read_csv("/home/niharika/Desktop/ML_Project/student/data.csv")

df.drop(['G1', 'G2'], axis=1, inplace=True)
df = df.drop(['school', 'famsize', 'reason',
        'guardian', 'famsup',  'activities', 'nursery',
        ], axis=1)



#0 stands for F and 1 stands for M
df['sex'] = df['sex'].apply(lambda x: 0 if x == 'F' else 1)
# 0 stands for U and 1 stands for R. [U=Urban, R=Rural]
df['address'] = df['address'].apply(lambda x: 0 if x == 'U' else 1)
# LE3 = Less than 3. [0], GE3 = Greater than 3.[1]
df['Pstatus'] = df['Pstatus'].apply(lambda x: 0 if x == 'T' else 1)
# 0 = no and 1 = yes
df['paid'] = df['paid'].apply(lambda x: 0 if x == 'no' else 1)
# 0 = no and 1 = yes
df['higher'] = df['higher'].apply(lambda x: 0 if x == 'no' else 1)
df['internet'] = df['internet'].apply(lambda x: 0 if x == 'no' else 1)
df['romantic'] = df['romantic'].apply(lambda x: 0 if x == 'no' else 1)
df['Mjob'] = df['Mjob'].apply(lambda x: 0 if x == 'at_home' else (1 if x=='health' else (2 if x=='other' else (3 if x=='services' else 4) )))
df['Fjob'] = df['Fjob'].apply(lambda x: 0 if x == 'at_home' else (1 if x=='health' else (2 if x=='other' else (3 if x=='services' else 4) )))
df['schoolsup'] = df['schoolsup'].apply(lambda x: 0 if x == 'no' else 1)

df['grade_status'] = df['G3'].apply(lambda x: 'Fail' if x < 12 else 'Pass')
df['grade_status'] = df['grade_status'].apply(lambda x: 0 if x == 'Fail' else 1)

df_concat = df

'''
df_Mjob = pd.get_dummies(df['Mjob']).iloc[:, 1:]
df_Fjob = pd.get_dummies(df['Fjob']).iloc[:, 1:]


df_concat = pd.concat([df_Mjob,df_Fjob,df], axis=1)
'''
X = df_concat.iloc[:, :-2].values
y = df_concat.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state =0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 23))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
classifier.save('ANN_student(2).model')
'''
new_prediction = classifier.predict(sc.transform(np.array([[0,15,0,0,1,1,0,3,1,2,0,1,0,1,1,0,4,3,2,2,3,3,6]])))
if (new_prediction > 0.5) :
    print("Fail")
else:
    print("Pass")
'''

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


