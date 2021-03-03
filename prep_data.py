# prep_data will prepare the data for future uses

import numpy as np
import pandas as pd
import seaborn as sns

#--------------------------------------------------------------------------------
# Read train.csv file
trainData = pd.read_csv("./data/train.csv")

# Print head
#print("\n", trainData.head())

# Print info
#print("\n", trainData.info())

# Print column names
#colNames = trainData.columns.values.tolist()
#print(colNames)

# Print data types
#print("\n", trainData.dtypes)

# Print na values
#print("\nNull vaues are:\n", trainData.isna().sum())

#--------------------------------------------------------------------------------

# Store id: passenger (fname + lname) pairs
#passengerId = {i+1: trainData.iloc[i, 3] for i in range(len(trainData["PassengerId"]))} -> less elegant (hard-coded index in .iloc)
#passengerId = {trainData.iloc[i, :]["PassengerId"]: trainData.iloc[i, :]["Name"] for i in range(len(trainData["PassengerId"]))}
#print(passengerId)

# Remove "Name" column
trainDataNoName = trainData.drop(labels="Name", axis=1)
#print(trainDataNoName.head())
#print(trainDataNoName.info())

# Export NoName for baseline model:
#trainDataNoName.to_csv("./data/baseline.csv", index=True) --> ALREADY RAN

# Print new colums names
colNoNames = trainDataNoName.columns.values.tolist()
print(colNoNames)