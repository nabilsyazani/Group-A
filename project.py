import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics

bbatch = 0
bepoch = 0
baccuracy = 0
minaccuracy = 1

# To remove the scientific notation from numpy arrays
np.set_printoptions(suppress=True)

df = pd.read_csv('C:/Uni/UTeM/SEM 4/NN Neural Network/Project/engine_data.csv')

# Separate Target Variable and Predictor Variables
TargetVariable = ['Engine_Condition']
Predictors = ['Engine_rpm', 'Lub_oil_pressure', 'Fuel_pressure',
              'Coolant_pressure', 'lub_oil_temp', 'Coolant_temp']

X = df[Predictors].values
y = df[TargetVariable].values

# Standardization of data
PredictorScaler = StandardScaler()

# Storing the fit object for later reference
PredictorScalerFit = PredictorScaler.fit(X)

# Generating the standardized values of X and y
X = PredictorScalerFit.transform(X)

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Quick sanity check with the shapes of Training and Testing datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

classifier = Sequential()
# Defining the Input layer
classifier.add(Dense(units=7, input_dim=6, kernel_initializer='uniform', activation='relu'))
# Defining the Hidden layer
classifier.add(Dense(units=4, kernel_initializer='uniform', activation='relu'))
# Defining the Output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fitting the Neural Network on the training data
survivalANN_Model = classifier.fit(X_train, y_train, batch_size=10, epochs=10, verbose=1)

# Defining a function for finding best hyperparameters
def FunctionFindBestParams(X_train, y_train):
    # Defining the list of hyper parameters to try
    TrialNumber = 0
    batch_size_list = [5, 10, 15, 20]
    epoch_list = [5, 10, 50, 100]

    SearchResultsData = pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])

    for batch_size_trial in batch_size_list:
        for epochs_trial in epoch_list:
            TrialNumber += 1

            # Creating the classifier ANN model
            classifier = Sequential()
            classifier.add(Dense(units=7, input_dim=6, kernel_initializer='uniform', activation='relu'))
            classifier.add(Dense(units=4, kernel_initializer='uniform', activation='relu'))
            classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
            classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            survivalANN_Model = classifier.fit(X_train, y_train, batch_size=batch_size_trial, epochs=epochs_trial,
                                               verbose=0)
            # Fetching the accuracy of the training
            Accuracy = survivalANN_Model.history['accuracy'][-1]

            # printing the results of the current iteration
            print(TrialNumber, 'Parameters:', 'batch_size:', batch_size_trial, '-', 'epochs:', epochs_trial,
                  'Accuracy:', Accuracy)

            global bbatch
            global bepoch
            global baccuracy
            global minaccuracy

            if float(Accuracy) < float(minaccuracy):
                minaccuracy = Accuracy

            if float(Accuracy) > float(baccuracy):
                bbatch = batch_size_trial
                bepoch = epochs_trial
                baccuracy = Accuracy

            SearchResultsData = pd.concat([SearchResultsData, pd.DataFrame(data=[[TrialNumber, 'batch_size' +
                                                                                  str(batch_size_trial) + '-' + 'epoch' +
                                                                                  str(epochs_trial), Accuracy]],
                                                                           columns=['TrialNumber', 'Parameters',
                                                                                    'Accuracy'])])
    return (SearchResultsData)


# Calling the function
ResultsData = FunctionFindBestParams(X_train, y_train)
# Printing the best parameter
print(ResultsData.sort_values(by='Accuracy', ascending=False).head(1))
# Visualizing the results
plt.figure(figsize=(12, 5))
plt.plot(ResultsData.Parameters, ResultsData.Accuracy, label='Accuracy')
plt.xticks(rotation=20)
plt.ylim([round(minaccuracy*0.8, 2), round(baccuracy*1.2, 2)])
plt.show()


# Training the model with best hyperparameters
classifier.fit(X_train, y_train, batch_size=bbatch, epochs=bepoch, verbose=1)

# Predictions on testing data
Predictions = classifier.predict(X_test)
# Scaling the test data back to original scale
Test_Data = PredictorScalerFit.inverse_transform(X_test)
# Generating a data frame for analyzing the test data
TestingData = pd.DataFrame(data=Test_Data, columns=Predictors)
TestingData['Condition'] = y_test
TestingData['PredictedProb'] = Predictions


# Defining the probability threshold
def probThreshold(inpProb):
    if inpProb > 0.5:
        return (1)
    else:
        return (0)


# Generating predictions on the testing data by applying probability threshold
TestingData['Predicted'] = TestingData['PredictedProb'].apply(probThreshold)
print(TestingData.head())
print('\nTesting Accuracy Results: ')
print(metrics.classification_report(TestingData['Condition'], TestingData['Predicted']))
print(metrics.confusion_matrix(TestingData['Condition'], TestingData['Predicted']))