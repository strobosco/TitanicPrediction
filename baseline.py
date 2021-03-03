# Create simple baseline model

import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Load train data
trainData = pd.read_csv("./data/baseline.csv", delimiter=',', sep=r', ')

# Load test data
#testData = pd.read_csv("./data/test.csv", delimiter=',', sep=r', ')

# Define LABEL and FEATURES
LABEL = ["Survived"]
FEATURES = ["Pclass", "Sex"]

# One-Hot Encode "Sex"
trainData["Sex"] = trainData["Sex"].map({"male": 0, "female": 1})

# One-Hot Encode "Pclass"
PclassDummies = pd.get_dummies(trainData["Pclass"], prefix="Pclass")
trainData = pd.concat([trainData, PclassDummies], axis=1)
trainData.drop("Pclass", axis=1)

# Split data
train_dataset = trainData.sample(frac=0.8, random_state=0)
test_dataset = trainData.drop(train_dataset.index)

# Normalization layer
featuresArray = np.array(trainData[FEATURES])

normalizer = preprocessing.Normalization(input_shape=[2,])
normalizer.adapt(featuresArray)

# Define model
model = keras.Sequential([
    normalizer,
    layers.Dense(units=4, activation="relu"),
    layers.Dense(units=1, activation="sigmoid")
])
model.summary()

# Compile model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Fit model
model.fit(
    train_dataset[FEATURES],
    train_dataset[LABEL],
    epochs=100,
    validation_split=0.2
)

# Evaluate model
loss, acc = model.evaluate(
    test_dataset[FEATURES],
    test_dataset[LABEL]
)

print("\nTest accuracy: ", acc) # Final baseline accuracy: 0.7191011309623718