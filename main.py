# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC


# simple function to find outliers for a feature
def find_outliers(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    floor = q1 - 1.5 * iqr
    ceiling = q3 + 1.5 * iqr
    outlier_indices = list(x.index[(x < floor) | (x > ceiling)])
    outlier_values = list(x[outlier_indices])
    return outlier_indices


# Iterates through each numerical column and pulls the indices of the outliers and drops games containing outliers
def remove_outliers(x):
    indices = []
    for c in x.columns:
        if not x[c].map(type).eq(str).any():
            if not c == "GAME_ID" or c == "GAME_DATE":
                indices += find_outliers(x[c])
    x = x.drop(indices)
    return x


# function that takes a teams data and fills any missing values in numerical columns for a dataframe
def clean_team(x):
    # separate numerical features and categorical features
    categorical_columns = []
    numeric_columns = []
    for c in x.columns:
        if x[c].map(type).eq(str).any():
            categorical_columns.append(c)
        else:
            numeric_columns.append(c)

    # create two dataframes to hold the two types
    data_numeric = x[numeric_columns]
    data_categorical = pd.DataFrame(x[categorical_columns])

    # replace missing values in numerical columns with median and then add the two types back together
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    data_numeric = pd.DataFrame(imp.fit_transform(data_numeric), columns=data_numeric.columns, index=data_numeric.index)
    x = pd.concat([data_numeric, data_categorical], axis=1)
    return x


if __name__ == '__main__':
    # read in file, convert outcome to numeric and select a specific teams data.
    # Check that data for missing values
    # pass data to cleaning function and outlier removal then split the clean data into an input set and an output set
    df = pd.read_csv('Final_dataset.csv')
    df['WL_HOME'] = [0 if x == 'L' else 1 for x in df['WL_HOME']]
    x = df.loc[(df['TEAM_ABBREVIATION_HOME'] == 'MIN') | (df['TEAM_ABBREVIATION_AWAY'] == 'MIN')]
    print(x.isnull().sum())
    x = clean_team(x)
    x = remove_outliers(x)
    teamInputFrame = x.drop('WL_HOME', 1)
    teamOutputFrame = x.WL_HOME
