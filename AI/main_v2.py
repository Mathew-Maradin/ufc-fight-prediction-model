# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_excel("ex_raw_total_fight_data.xlsx", skiprows=1)
# print(df)

features = df.drop('R_SIG_STR.', axis=1)
features = features.drop('B_SIG_STR.', axis=1)
features = features.drop('R_TOTAL_STR.', axis=1)
features = features.drop('B_TOTAL_STR.', axis=1)
features = features.drop('R_TD', axis=1)
features = features.drop('B_TD', axis=1)
features = features.drop('R_HEAD', axis=1)
features = features.drop('B_HEAD', axis=1)
features = features.drop('R_BODY', axis=1)
features = features.drop('B_BODY', axis=1)
features = features.drop('R_LEG', axis=1)
features = features.drop('B_LEG', axis=1)
features = features.drop('R_DISTANCE', axis=1)
features = features.drop('B_DISTANCE', axis=1)
features = features.drop('R_CLINCH', axis=1)
features = features.drop('B_CLINCH', axis=1)
features = features.drop('R_GROUND', axis=1)
features = features.drop('B_GROUND', axis=1)
features = features.drop('Winner', axis=1)
features = features.drop('last_round_time', axis=1)
features = features.drop('R_CTRL', axis=1)
features = features.drop('B_CTRL', axis=1)
features = features.drop('Format', axis=1)
features = features.drop('Referee', axis=1)
features = features.drop('date', axis=1)
features = features.drop('location', axis=1)
features = features.drop('Fight_type', axis=1)
features = features.drop('win_by', axis=1)
features = features.replace("---", 0)
# print(features)
print("features processed")

target = df['Winner']
# print(target)
print("target processed")

for i in range(7183):
    value = features.iloc[i, 0]
    target_val = target.iloc[i]

    if (value == target_val):
        target.iat[i] = 0
    else:
        target.iat[i] = 1

    features.iat[i, 0] = 0
    features.iat[i, 1] = 1

# print(target)
# print(features)

target = target.to_frame()

features = target.astype(float)
non_numeric_cols = features.select_dtypes(exclude=['int', 'float']).columns
non_numeric_values_exist = features[non_numeric_cols].apply(lambda x: pd.to_numeric(x, errors='coerce').isna().any()).any()

features = features.astype(float)
non_numeric_cols = features.select_dtypes(exclude=['int', 'float']).columns
non_numeric_values_exist = features[non_numeric_cols].apply(lambda x: pd.to_numeric(x, errors='coerce').isna().any()).any()






model = RandomForestClassifier()
model.fit(features, target)


# df_upcoming = pd.read_excel("upcoming-test.xlsx")
# #Get the number of upcoming fights
# num_upcoming_fights = len(df_upcoming)
# print(f"We are going to predict the winner of {num_upcoming_fights} fights.")

# predictions = model.predict(df_upcoming)
# df_upcoming['Predicted_Winner'] = predictions
# print(df_upcoming[['Fighter_1', 'Fighter_2', 'Predicted_Winner']])

print("success")