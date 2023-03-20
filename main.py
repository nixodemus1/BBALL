# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict


# simple function to find outliers for a feature using tukey method
# @param x: a  column of data from the dataset
def find_outliers(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    floor = q1 - 1.5 * iqr
    ceiling = q3 + 1.5 * iqr
    outlier_indices = list(x.index[(x < floor) | (x > ceiling)])
    outlier_values = list(x[outlier_indices])
    return outlier_indices, outlier_values


# Iterates through each numerical column and pulls the indices of the outliers and drops games containing outliers
# @param x: a dataframe containing the teams games
def remove_outliers(x):
    indices = []
    for c in x.columns:
        if not x[c].map(type).eq(str).any():
            if not c == "GAME_ID" or c == "GAME_DATE":
                indices += find_outliers(x[c])[0]
    x = x.drop(indices)
    return x


# function that takes a teams data and fills any missing values in numerical columns for a dataframe
# @param x: a dataframe which may contain missing values
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


# creates a subset of data containing all of a specific teams games
# @param team: The abbreviation for the team to be pulled
# @param games: a dataframe containing the set of all games
def pull_team(team, games):
    teams_games = games.loc[(games['TEAM_ABBREVIATION_HOME'] == team) |
                            (games['TEAM_ABBREVIATION_AWAY'] == team)]
    print('Number of games')
    print(len(teams_games))
    print('Missing Values', games.isnull().sum())
    return teams_games


if __name__ == '__main__':
    # read in file, convert outcome to numeric and select a specific teams data.
    # Check that data for missing values
    # pass data to cleaning function and outlier removal then split the clean data into an input set and an output set
    df = pd.read_csv('data/Final_dataset.csv')
    df['WL_HOME'] = [0 if x == 'L' else 1 for x in df['WL_HOME']]
    x = pull_team("MIN", df)
    x = clean_team(x)
    x = remove_outliers(x)
    teamIF = x.drop(['WL_HOME', 'GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION_HOME.1', 'TEAM_ABBREVIATION_HOME',
                     'TEAM_ABBREVIATION_AWAY'], axis=1)
    teamOF = x.WL_HOME

    # JUST FOR TEST PURPOSES
    g = pull_team("GSW", df)
    g = clean_team(g)
    g = remove_outliers(g)
    gswIF = g.drop(['WL_HOME', 'GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION_HOME.1', 'TEAM_ABBREVIATION_HOME',
                    'TEAM_ABBREVIATION_AWAY'], axis=1)
    gswOF = g.WL_HOME
    gswIF_train, gswIF_test, gswOF_train, gswOF_test = train_test_split(gswIF, gswOF, test_size=.25)
    scale = MinMaxScaler()
    scale.fit(gswIF_train)
    gswIF_train_scale = scale.transform(gswIF_train)
    gswIF_test_scale = scale.transform(gswIF_test)

    league = clean_team(df)
    leagueIF = league.drop(['WL_HOME', 'GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION_HOME.1', 'TEAM_ABBREVIATION_HOME',
                            'TEAM_ABBREVIATION_AWAY'], axis=1)
    leagueOF = league.WL_HOME
    leagueIF_train, leagueIF_test, leagueOF_train, leagueOF_test = train_test_split(leagueIF, leagueOF, test_size=.25)
    scale = MinMaxScaler()
    scale.fit(leagueIF_train)
    leagueIF_train_scale = scale.transform(leagueIF_train)
    leagueIF_test_scale = scale.transform(leagueIF_test)
    # end of gsw test

    # split into test and training set and scale using minmax
    teamIF_train, teamIF_test, teamOF_train, teamOF_test = train_test_split(teamIF, teamOF, test_size=.25)
    scale = MinMaxScaler()
    scale.fit(teamIF_train)
    teamIF_train_scale = scale.transform(teamIF_train)
    teamIF_test_scale = scale.transform(teamIF_test)

    # cross validation
    teamIF_train.shape, teamOF_train.shape
    teamIF_test.shape, teamOF_test.shape

    # initialize svm with rbf kernel and fit it to the dataset
    our_svm = SVC(C=1.0,kernel='rbf',gamma='scale')
    our_svm.fit(teamIF_train_scale, teamOF_train)
    pred = our_svm.predict(teamIF_test_scale)

    #plot the model
    loss_train = history['train_loss']
    loss_val = history['val_loss']
    epochs = range(1,35)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    #get the confidence of each prediction and then standardize it into a percentage
    confidence = our_svm.decision_function(teamIF_test_scale)
    max_pred = 0
    percent = np.array([])
    if np.max(confidence) < abs(np.min(confidence)):
        max_pred = abs(np.min(confidence))
    else:
        max_pred = np.max(confidence)
    for i in confidence:
        per = (i/max_pred)*100
        percent = np.append(percent, per)

    # estimating accuracy, by computing the score 5 times
    scores = cross_val_score(our_svm, teamIF, teamOF, cv=5)

    # display results
    print(classification_report(teamOF_test, pred))
    print(our_svm.score(teamIF_test, teamOF_test))

    # cross accuracy and standard deviation results
    print("score has %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    data = {'predicted result': pred, 'actual result': teamOF_test, 'confidence in prediction': percent}
    results = pd.DataFrame(data)
    print(results)