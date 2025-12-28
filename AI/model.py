import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_logistic_regression(X, y):
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    # Evaluate
    y_pred = lr.predict(X_test)
    print("\n=== Logistic Regression Baseline ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return lr

def train_random_forest(X, y):
    """
    Train and evaluate a Random Forest classifier.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize model
    rf = RandomForestClassifier(
        n_estimators=200,      # number of trees
        max_depth=10,          # limit depth to prevent overfitting
        random_state=42,
        class_weight="balanced" # handles class imbalance
    )
    
    # Train
    rf.fit(X_train, y_train)
    
    # Predict
    y_pred = rf.predict(X_test)
    
    # Evaluate
    print("=== Random Forest ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    return rf
def buildFighterStats(path="../data/completed_fights.xlsx"):
    fights = pd.read_excel(path)

    # Keep only numerical features
    stats = fights.drop(
        ['R_fighter', 'B_fighter', 'Referee', 'Details', 'date',
         'location', 'Fight_type', 'Winner', 'win_by',
         'last_round', 'last_round_time', 'Format', 'R_CTRL', 'B_CTRL'],
        axis=1,
        errors="ignore"
    )

    for col in stats.columns:
        stats[col] = pd.to_numeric(stats[col], errors="coerce")

    # Collect Red corner stats
    R_stats = fights[['R_fighter']].join(stats.filter(like="R_"))
    R_stats.rename(columns={'R_fighter': 'fighter'}, inplace=True)
    R_stats.columns = ['fighter'] + [c.replace("R_", "") for c in R_stats.columns[1:]]

    # Collect Blue corner stats
    B_stats = fights[['B_fighter']].join(stats.filter(like="B_"))
    B_stats.rename(columns={'B_fighter': 'fighter'}, inplace=True)
    B_stats.columns = ['fighter'] + [c.replace("B_", "") for c in B_stats.columns[1:]]

    # Combine & average
    all_stats = pd.concat([R_stats, B_stats])
    fighter_avgs = all_stats.groupby("fighter").mean()

    return fighter_avgs

def buildHypotheticalFight(fighterA, fighterB, fighter_avgs, feature_columns):
    # Get avg stats for each fighter
    statsA = fighter_avgs.loc[fighterA]
    statsB = fighter_avgs.loc[fighterB]

    # Compute R - B difference
    diff = statsA - statsB

    # Build DataFrame with EXACT training feature columns
    X = pd.DataFrame([diff], columns=fighter_avgs.columns)

    # Reindex to match training features (order + names)
    X = X.reindex(columns=[c.replace("R_", "") for c in feature_columns], fill_value=0)
    X.columns = feature_columns

    return X


def predictHypotheticalFight(fighterA, fighterB, model, fighter_avgs, feature_columns):
    X = buildHypotheticalFight(fighterA, fighterB, fighter_avgs, feature_columns)

    probs = model.predict_proba(X)[0]
    pred = model.predict(X)[0]

    return {
        "R_fighter": fighterA,
        "B_fighter": fighterB,
        "Prob_R_Wins": float(probs[1]),
        "Prob_B_Wins": float(probs[0]),
        "Predicted_Winner": fighterA if pred == 1 else fighterB
    }

def loadDF():
    import pandas as pd

    fights = pd.read_excel('../data/completed_fights.xlsx')

    # Prepare target variable
    Y = fights["Winner"].str.strip() == fights["R_fighter"].str.strip()
    y = Y.astype(int).reset_index(drop=True)

    # Select numerical stats only
    numerical_statistics = fights.drop(
        ['R_fighter', 'B_fighter', 'Referee', 'Details', 'date',
         'location', 'Fight_type', 'Winner', 'win_by',
         'last_round', 'last_round_time', 'Format', 'R_CTRL', 'B_CTRL'],
        axis=1
    )

    for col in numerical_statistics.columns:
        numerical_statistics[col] = pd.to_numeric(numerical_statistics[col], errors="coerce")

    # Compute difference: R_ stats - B_ stats
    R_cols = numerical_statistics.filter(like="R_").columns
    B_cols = numerical_statistics.filter(like="B_").columns

    X = pd.DataFrame()
    for r_col, b_col in zip(R_cols, B_cols):
        X[r_col] = numerical_statistics[r_col] - numerical_statistics[b_col]

    # Replace any NaN or infinite values with 0
    X.replace([float('inf'), -float('inf')], 0, inplace=True)
    X.fillna(0, inplace=True)

    return X, y

def predictEvent(model, fighter_avgs, feature_columns):
    fights = pd.read_excel('../data/upcoming_fights.xlsx')
    fights.head()

    winners = []

    for index, row in fights.iterrows():
        try:
            result = predictHypotheticalFight(row['R_fighter'], row['B_fighter'], model, fighter_avgs, feature_columns)

            if row['R_fighter'] == result['Predicted_Winner']:
                winners.append([result['Predicted_Winner'], result['Prob_R_Wins']])
            else:
                winners.append([result['Predicted_Winner'], result['Prob_B_Wins']])

        except:
            continue

    
    print("\nMoney Line locks for the boys")
    
    for winner in winners:
        print(str(winner[0]), 'probability: ' + str(round(winner[1])*100) + '%')



def __init__():
    print("Preprocessing data")
    X, y = loadDF()

    # lin_regression_model = train_logistic_regression(X, y)
    rf_model = train_random_forest(X, y)

    # Build fighter stats
    fighter_avgs = buildFighterStats()

    # Save the feature columns from training
    feature_columns = X.columns

    predictEvent(rf_model, fighter_avgs, feature_columns)


__init__()