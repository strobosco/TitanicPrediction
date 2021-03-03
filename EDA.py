# Perform EDA on passenger data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Load train and test data
trainData = pd.read_csv("./data/train.csv")
testData = pd.read_csv("./data/test.csv")

# Head, info, and describe
print("\nTrain Head:\n", trainData.head())
print("\nTrain Info:\n", trainData.info())
print("\nTrain Describe:\n", trainData.describe())

print("\nTest Head:\n", testData.head())
print("\nTest Info:\n", testData.info())
print("\nTest Describe:\n", testData.describe())

# Survival chance overall
plt.close()
sns.countplot(x="Survived", data=trainData)
plt.suptitle("The percentage of survivors is: {}".format(trainData.Survived.sum()/trainData.Survived.count()))
plt.show()


# Survival chance & sex
res = trainData.groupby(["Survived", "Sex"])["Survived"].count()
print("\nSurvival & Sex:\n", res)

plt.close()
sns.catplot(x="Sex", col="Survived", kind="count", data=trainData)

femalePercentage = trainData[trainData.Sex == "female"].Survived.sum() / trainData[trainData.Sex == "female"].Survived.count()
malePercentage = trainData[trainData.Sex == "male"].Survived.sum() / trainData[trainData.Sex == "male"].Survived.count()
plt.suptitle("Percentage of women that survived: {:.2f} \nof men: {:.2f}".format(femalePercentage, malePercentage), x=0.60, y=0.75)

plt.show()


# Passenger class
crosstab = pd.crosstab(trainData.Pclass, trainData.Survived, margins=True)
print(crosstab)
print("\n% of survivals in") 
print("\nPclass=1 : ", trainData.Survived[trainData.Pclass == 1].sum()/trainData[trainData.Pclass == 1].Survived.count())
print("\nPclass=2 : ", trainData.Survived[trainData.Pclass == 2].sum()/trainData[trainData.Pclass == 2].Survived.count())
print("\nPclass=3 : ", trainData.Survived[trainData.Pclass == 3].sum()/trainData[trainData.Pclass == 3].Survived.count())

# Save baseline.csv
#trainData.to_csv("./data/baseline.csv", index=True) ---> ALREADY RAN