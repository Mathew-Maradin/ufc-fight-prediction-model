import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')
from preprocessing import FightProcessor
from typing import Literal, TypeAlias
import xgboost as xgb
import numpy as np

FightType: TypeAlias = Literal["UPCOMING", "COMPLETED"]        

class Model:
    def __init__(self):
        self.processor = FightProcessor()

    def create_training_data(self):
        fight_data = self.processor.process_data("COMPLETED")
        fight_data = fight_data.dropna(subset=['Winner'])

        X = fight_data[[
                'AVG_DIFF_B_KD',
                'AVG_DIFF_B_SIG_STR',
                'AVG_DIFF_B_SIG_STR_pct',
                'AVG_DIFF_B_TOTAL_STR',
                'AVG_DIFF_B_TD',
                'AVG_DIFF_B_TD_pct',
                'AVG_DIFF_B_SUB_ATT',
                'AVG_DIFF_B_REV',
                'AVG_DIFF_B_CTRL',
                'AVG_DIFF_B_HEAD',
                'AVG_DIFF_B_BODY',
                'AVG_DIFF_B_LEG',
                'AVG_DIFF_B_DISTANCE',
                'AVG_DIFF_B_CLINCH',
                'AVG_DIFF_B_GROUND']]
        
        y = fight_data['Winner']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return [fight_data, X_train, X_test, y_train, y_test]

    def train_rf(self, fight_data, X_train, X_test, y_train, y_test):

        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:\n", classification_rep)

    def train_xg(self, fight_data, X_train, X_test, y_train, y_test):
        xgb_train = xgb.DMatrix(X_train, y_train, enable_categorical=True)
        xgb_test = xgb.DMatrix(X_test, y_test, enable_categorical=True)

        params = {
            'objective': 'binary:logistic',
            'max_depth': 3,
            'learning_rate': 0.1,
        }
        n=50
        model = xgb.train(params=params,dtrain=xgb_train,num_boost_round=n)

        preds = model.predict(xgb_test)
        preds = np.round(preds)
        accuracy= accuracy_score(y_test,preds)

        classification_rep = classification_report(y_test, preds)

        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:\n", classification_rep)