import time, pathlib, os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.exceptions import NotFittedError
from abc import ABC, abstractmethod
from src import Preprocessor

filePath = pathlib.Path(__file__).parent.resolve()

class Classifier(ABC):
    def __init__(self, preprocessor: Preprocessor, class_column: str, train_split: float = 2/3, verbose=False, random_seed=int(time.time())):
        print(f"[CLASSIFIER: {self.__class__.__name__}]")
        self._preprocessor = preprocessor
        self._class_column = class_column
        self._verbose = verbose
        self._model = None
        self._trained = False
        self._random_seed = random_seed
        print("[CLASSIFIER] seed:", self.random_seed)
        
        # Split the labeled training data into a train and test set
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self._preprocessor.train_data.loc[:, self._preprocessor.train_data.columns != class_column],
            self._preprocessor.train_data[class_column],
            train_size=train_split,
            random_state=self._random_seed
        )
        
        # Evaluation metrics
        self._train_accuracy = None
        self._test_accuracy = None
        self._roc_auc = None
        self._precision = None
        self._recall = None
    
    def predict(self):
        # Fill out the predictions template
        templateFilePath = (filePath / "../data/predictions_template.csv").resolve()
        outputFilePath = (filePath / "../processed_data/predictions.csv").resolve()
        templates = pd.read_csv(templateFilePath)
        
        predictions = pd.Series(self._model.predict(self._preprocessor.test_predictions_data))
        templates[self._preprocessor._label_column] = predictions
        
        templates.to_csv(outputFilePath, index=False)
    
    def cross_validate(self, param_grid, score_metric="accuracy"):
        print(f"[CLASSIFIER] running cross validation on parameter grid (GridSearchCV), optimizing for {score_metric} score")
        gscv = GridSearchCV(self._model, param_grid=param_grid, scoring=score_metric, n_jobs=-1)
        gscv.fit(self._X_train, self._y_train)
        print("\tbest estimator:", gscv.best_estimator_)
        print("\tbest parameters:", gscv.best_params_)
        print("\tbest accuracy score:", gscv.best_score_)
        return gscv.best_params_
        
    def _train_model(self):
        if self._verbose: print("[CLASSIFIER] training model")
        self._model.fit(self._X_train, self._y_train)
        if self._verbose: print("\tdetermining training set prediction accuracy score")
        # Make predictions on the training data to get training accuracy
        train_preds = self._model.predict(self._X_train)
        self._train_accuracy = accuracy_score(self._y_train, train_preds)
        
        self._trained = True
        
    def _evaluate_model(self):
        if not self._trained:
            self._train_model()
        if self._verbose:
            print("[CLASSIFIER] evaluating model")
            print("\trunning predictions on test split")
        y_predictions = self._model.predict(self._X_test)
        if self._verbose: print("\tdetermining test set accuracy, auc, precision and recall score")
        self._test_accuracy = accuracy_score(self._y_test, y_predictions)
        self._roc_auc = roc_auc_score(self._y_test, y_predictions)
        self._precision = precision_score(self._y_test, y_predictions)
        self._recall = recall_score(self._y_test, y_predictions)
    
    @property
    def model(self):
        if self._model is None:
            print("[CLASSIFIER] no model instantiated in concrete class")
        return self._model

    @property
    def random_seed(self):
        return self._random_seed
    
    @property
    def train_accuracy(self):
        if self._train_accuracy is None:
            self._train_model()
        return self._train_accuracy
    
    @property
    def test_accuracy(self):
        if self._test_accuracy is None:
            self._evaluate_model()
        return self._test_accuracy
    
    @property
    def roc_auc(self):
        if self._roc_auc is None:
            self._evaluate_model()
        return self._roc_auc
    
    @property
    def precision(self):
        if self._precision is None:
            self._evaluate_model()
        return self._precision
    
    @property
    def recall(self):
        if self._recall is None:
            self._evaluate_model()
        return self._recall
    
class DecisionTree(Classifier):
    def __init__(self, preprocessor: Preprocessor, class_column: str, train_split: float = 2/3, model_params: dict={}, verbose=False, random_seed=int(time.time())):
        super().__init__(
            preprocessor=preprocessor, 
            class_column=class_column,
            train_split=train_split,
            verbose=verbose,
            random_seed=random_seed
        )
        self._model=DecisionTreeClassifier(
            random_state=self._random_seed,
            **model_params
        )
    
class RandomForest(Classifier):
    def __init__(self, preprocessor: Preprocessor, class_column: str, train_split: float = 2/3, model_params: dict={}, verbose=False, random_seed=int(time.time())):
        super().__init__(
            preprocessor=preprocessor, 
            class_column=class_column,
            train_split=train_split,
            verbose=verbose,
            random_seed=random_seed
        )
        self._model=RandomForestClassifier(
            random_state=self._random_seed,
            **model_params
        )
        
    @property
    def importances(self):
        try: 
            # source: https://www.geeksforgeeks.org/machine-learning/feature-importance-with-random-forests/
            importances = self.model.feature_importances_
            feature_imp_df = pd.DataFrame({'Feature': self._X_train.columns.values, 'Importance': importances}).sort_values(
                'Importance', ascending=False)
            
            # create plot of importances
            plotPath = (filePath / "../processed_data/feature_importances.png").resolve()
            plt.figure(figsize=(10, 8))
            plt.barh(feature_imp_df['Feature'][::-1], feature_imp_df['Importance'][::-1])
            plt.title('Feature Importances', fontsize=18, fontweight="bold")
            plt.xlabel('Importance Score', fontsize=12, fontweight="bold")
            plt.ylabel('Feature', fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig(plotPath)
            return plotPath
        except NotFittedError:
            self._train_model()
            return self.importances
    
class GaussianNaiveBayes(Classifier):
    def __init__(self, preprocessor: Preprocessor, class_column: str, train_split: float = 2/3, model_params: dict={}, verbose=False, random_seed=int(time.time())):
        super().__init__(
            preprocessor=preprocessor, 
            class_column=class_column,
            train_split=train_split,
            verbose=verbose,
            random_seed=random_seed
        )
        self._model=GaussianNB(**model_params)
    
class KNearestNeighbors(Classifier):
    def __init__(self, preprocessor: Preprocessor, class_column: str, train_split: float = 2/3, model_params: dict={}, verbose=False, random_seed=int(time.time())):
        super().__init__(
            preprocessor=preprocessor, 
            class_column=class_column,
            train_split=train_split,
            verbose=verbose,
            random_seed=random_seed
        )
        self._model=KNeighborsClassifier(**model_params)
