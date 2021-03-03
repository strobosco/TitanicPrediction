# Model 01 - started 01/03/2021 | completed 01/03/2021
# Model accuracy: 0.8089887499809265

import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing as preprop

#--------------------------------------------------------------

def loadData():
    return pd.read_csv("./data/train.csv")

def preprocessing():

    trainData = loadData()

    RELCOLS = ["Survived", "Sex", "Pclass", "Age", "Embarked"]
    trainData = trainData[RELCOLS]

    # One-Hot Encode "Sex"
    trainData["Sex"] = trainData["Sex"].map({"male": 0, "female": 1})

    # One-Hot Encode "Embarked"
    embarkedDummies = pd.get_dummies(trainData["Embarked"], prefix="Embarked")
    trainData = pd.concat([trainData, embarkedDummies], axis=1)
    trainData.drop("Embarked", axis=1, inplace=True)

    # One-Hot Encode "Pclass"
    PclassDummies = pd.get_dummies(trainData["Pclass"], prefix="Pclass")
    trainData = pd.concat([trainData, PclassDummies], axis=1)
    trainData.drop("Pclass", axis=1, inplace=True)

    # Replace N/A "Age" values with median
    trainData["Age"].fillna(trainData["Age"].mean(), inplace=True)

    print(trainData.describe())

    # Split data
    trainingData = trainData.sample(frac=0.8, random_state=0)
    testingData = trainData.drop(trainingData.index)

    return trainingData, testingData

def model():

    #FEATURES = ["Sex", "Pclass", "Age", "Embarked"]
    #LABEL = ["Survived"]

    training, testing = preprocessing()

    FEATURES = training.columns.values[1:]
    LABEL = training.columns.values[0]
    print(FEATURES)
    print(LABEL)

    # Define normalizer
    inputShape = np.array(training[FEATURES])
    normalizer = preprop.Normalization()
    normalizer.adapt(inputShape)

    # Define model
    model = keras.Sequential([
        normalizer,
        layers.Dense(units=8, activation="relu"),
        layers.Dense(units=8, activation="relu"),
        layers.Dense(units=1, activation="sigmoid")
    ])
    model.summary()

    # Compile model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"]
    )

    # Fit model
    model.fit(
        training[FEATURES],
        training[LABEL],
        epochs=100,
        validation_split=0.2
    )

    # Evaluate model
    loss, acc = model.evaluate(
        testing[FEATURES],
        testing[LABEL]
    )

    print("\nTest accuracy: ", acc)

    model.save("./models/model01.h5", overwrite=True, include_optimizer=True)



if __name__ == "__main__":
    
    model()