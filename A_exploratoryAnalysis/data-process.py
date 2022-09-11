import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

columns = [
    "age",
    "workClass",
    "financialWeight",
    "education",
    "educationNum",
    "maritalStatus",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capitalGain",
    "capitalLoss",
    "hoursPerWeek",
    "nativeCountry",
    "incomeTarget",
]


print(len(columns))

df = pd.read_csv("../data/adult-data.csv", names=columns)


traindata, valdata = train_test_split(df, test_size=0.3, random_state=1)
train_df = traindata.reset_index(drop=True)
val_df = valdata.reset_index(drop=True)


train_df.to_csv("../data/adult-train.csv", index=False)
val_df.to_csv("../data/adult-val.csv", index=False)

print("complete", train_df.shape, val_df.shape)
