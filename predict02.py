# Predictions for model02

import pandas as pd
import tensorflow as tf

#--------------------------------------------------------------

# Load model and data
def loading():
    model = tf.keras.models.load_model("./models/model02.h5")
    testData = pd.read_csv("./data/test.csv")

    return model, testData

def processData():

    model, testData = loading()

    FEATURES = ["Sex", "Pclass", "Age", "Embarked"]
    testData = testData[FEATURES]

    # One-Hot Encodings:
    # One-Hot Encode "Sex"
    testData["Sex"] = testData["Sex"].map({"male": 0, "female": 1})

    # One-Hot Encode "Embarked"
    embarkedDummies = pd.get_dummies(testData["Embarked"], prefix="Embarked")
    testData = pd.concat([testData, embarkedDummies], axis=1)
    testData.drop("Embarked", axis=1, inplace=True)

    # One-Hot Encode "Pclass"
    PclassDummies = pd.get_dummies(testData["Pclass"], prefix="Pclass")
    testData = pd.concat([testData, PclassDummies], axis=1)
    testData.drop("Pclass", axis=1, inplace=True)

    # Replace N/A "Age" values with median
    testData["Age"].fillna(testData["Age"].median(), inplace=True)

    return model, testData

def predict():

    model, testData = processData()

    predictions = model.predict(
        testData
    )

    print("Predictions are:\n", predictions)

if __name__ == "__main__":

    predict()