# ANN
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

# dataset
data_set = pd.read_csv("D:\\DotPy\\Deep Learning\\lec11 ANN\\data\\Churn_Modelling.csv")
x = pd.get_dummies(data_set.iloc[:, 3:13], drop_first=True)
x = x.values
y = data_set.iloc[:, -1].values

# scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

# splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# building the first model

model = Sequential([
                    Dense(units=6, activation='relu'),  # hidden layer 1
                    Dense(units=6, activation='relu'),   # hidden layer 2
                    Dense(units=1, activation='sigmoid')  # output layer
])

#compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fitting the model
model.fit(x_train, y_train, batch_size=32, epochs=30)

#evaluating the model
from sklearn.metrics import confusion_matrix, accuracy_score
y_prediction = model.predict(x_test)
y_prediction = y_prediction>0.5
cm = confusion_matrix(y_test, y_prediction)
score = accuracy_score(y_test, y_prediction)
print(cm)
print("the accuracy of the model is:", score)

# using the model in new data observation
new_data = [400, 30, 2, 0, 1, 1, 1, 100000, 1, 0, 1]

new_prediction =model.predict(sc.transform([new_data]))
print("the new observation is:", new_prediction>0.5)


#####################################################################
#####################################################################

#building the second model

from sklearn.linear_model import LogisticRegression
model2 = LogisticRegression()
model2.fit(x_train, y_train)

#evaluating the second model

y_prediction2 = model2.predict(x_test)
y_prediction2 = y_prediction2>0.5
cm2 = confusion_matrix(y_test, y_prediction2)
score2 = accuracy_score(y_test, y_prediction2)
print(cm2)
print("the accuracy of the model is:", score2)

# using the model in new data observation
new_data = [400, 30, 2, 0, 1, 1, 1, 100000, 1, 0, 1]

new_prediction2 =model2.predict(sc.transform([new_data]))
print("the new observation is:", new_prediction2>0.5)


