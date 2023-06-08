import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }
            params={
                "Decision Tree": {
                    'criterion':['gini', 'entropy', 'log_loss'],
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,386]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic Regression":{},
                "XGBClassifier":{
                    'learning_rate':[.3,.5,.01,.1],
                    'n_estimators': [3,10,30]
                },
                "AdaBoost Classifier":{
                    'learning_rate':[.3,.5,.01,.1],
                    'n_estimators': [3,10,30]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
            



            
        except Exception as e:
            raise CustomException(e,sys)