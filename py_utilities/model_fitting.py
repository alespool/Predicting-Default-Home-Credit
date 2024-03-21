from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

class ModelFitter:
    def __init__(self):
        """
        Initialize the ModelFitter with a DataFrame.

        Parameters:
            df (pandas.DataFrame): The input DataFrame.
        """
        self.modelclasses = [
            ["random forest", RandomForestClassifier(random_state=42), {"clf__n_estimators": [100, 200], "max_depth": [1,2]}],
            ["gradient boosting", GradientBoostingClassifier(random_state=42), {"clf__learning_rate": [0.05, 0.1], "max_depth": [1,2]}],
            ["adaboost", AdaBoostClassifier(), {"clf__n_estimators": [100, 200] , "max_depth": [1,2]}],
       ]
        self.insights = []

    def split_data(self, X, y, test_size:float =0.25):
        """
        Splits the data into training and testing sets.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.
            test_size (float): The proportion of the data to be used for testing. Default is 0.25.

        Returns:
            X_train (array-like): The training features.
            X_test (array-like): The testing features.
            y_train (array-like): The training target variable.
            y_test (array-like): The testing target variable.
        """
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=test_size,
                                                            random_state=42,
                                                            stratify=y)
        return X_train, X_test, y_train, y_test

    def insert_sort(self,arr):
        """
        Sorts the given array in descending order based on the third element of each element.

        Parameters:
            arr (list): The array to be sorted.

        Returns:
            None
        """
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j][2] < key[2]:  # Compare scores
                j -= 1
            # Insert 'key' to the right of 'j'
            arr.insert(j + 1, arr.pop(i))
    
    def rescale_imbalanced_data(self, X, y, resample:str):
        """
        Rescales imbalanced data using the specified resampling strategy.

        Parameters:
            self: object
                The instance of the class.
            X: array-like, shape (n_samples, n_features)
                The input data.
            y: array-like, shape (n_samples,)
                The target labels.
            resample: str
                The resampling method, either 'oversample' or 'undersample'.
            strategy: float
                The resampling strategy.

        Returns:
            X_resampled: array-like, shape (n_samples_new, n_features)
                The resampled input data.
            y_resampled: array-like, shape (n_samples_new,)
                The resampled target labels.
        """
        
        if resample == 'oversample':
            resampler = ADASYN(random_state=42)
        elif resample == 'undersample':
            resampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = resampler.fit_resample(X, y)
        return X_resampled, y_resampled

    
    def automate_model_selection(self, X_train, y_train, X_test, y_test, scoring:str = 'precision', top_k=2):
        """
        Automates the model selection process by iterating over a list of model classes, fitting each model to the training data using GridSearchCV for hyperparameter tuning, and storing the best parameters and scores in a list of insights. Finally, the list of insights is sorted, and the best model is returned.

        Parameters:
            X_train (array-like): The training input samples.
            y_train (array-like): The target values for the training input samples.
            X_test (array-like): The test input samples.
            y_test (array-like): The target values for the test input samples.
            scoring (str): The scoring metric to use for hyperparameter tuning.
            top_k (int): Number of top models to include in the ensemble.

        Returns:
            tuple: A tuple containing the best model name, best parameters, and the best model itself.

        """
        X_train_small, X_val, y_train_small, y_val = self.split_data(X_train, y_train, test_size=0.2)

        best_model = None
        best_score = 0

        for modelname, model, params in self.modelclasses:  
            pipe = Pipeline(steps=[('clf', model)])
            grid = GridSearchCV(pipe, params, cv=3, n_jobs=-1, scoring=scoring)
            grid.fit(X_train_small, y_train_small)
            score = grid.score(X_val, y_val)
            best_params = grid.best_params_

            self.insights.append((modelname, best_params, score, grid.best_estimator_))

            if score > best_score:
                best_score = score
                best_model = grid.best_estimator_

        self.insert_sort(self.insights)

        best_model.fit(X_train, y_train)

        best_model_score = best_model.score(X_test, y_test)

        print("*" * 80)
        print(f"The best model is: {best_model}")
        print("*" * 80)
        print(f"Accuracy for the best model: {best_model_score * 100:.2f}%")

        final_3_models = self.insights[:3]
        print("Scores for the final 3 models:")
        for modelname, _, score, _ in final_3_models:
            print(f"{modelname}: {score*100:.2f}%")

        return best_model

    def evaluate_classification_confusion(self, model, X_train, y_train, X_test, y_test, use_train_predictions=True):
        """
        Evaluate a classification model using the provided data and predictions.
    
        Parameters:
            model: The trained classification model.
            X_train: Feature variables for training data.
            y_train: True labels for training data.
            X_test: Feature variables for testing data.
            y_test: True labels for testing data.
            use_train_predictions: If True, use training data predictions for the confusion matrix. Otherwise, use testing data predictions.
    
        Returns:
            dict: A dictionary containing confusion matrix.
        """
        if use_train_predictions:
            y_pred = model.predict(X_train)
        else:
            y_pred = model.predict(X_test)
    
        conf_matrix = confusion_matrix(y_train if use_train_predictions else y_test, y_pred)
    
        return conf_matrix

    def evaluate_classification_model(self, model, X, y, data_type='test'):
        """Create a classification report for metrics."""
        if data_type == 'train':
            y_pred = model.predict(X)
        else:
            y_pred = model.predict(X)
            
        class_report = classification_report(y, y_pred, output_dict=True)
        logging.info(f"{data_type.capitalize()} Classification Report: {class_report}")
        
        return class_report, y_pred
    
    def evaluate_classification_confusion(self, model, X_train, y_train, X_test, y_test, use_train_predictions=True):
        """
        Evaluate a classification model using the provided data and predictions.
    
        Parameters:
            model: The trained classification model.
            X_train: Feature variables for training data.
            y_train: True labels for training data.
            X_test: Feature variables for testing data.
            y_test: True labels for testing data.
            use_train_predictions: If True, use training data predictions for the confusion matrix. Otherwise, use testing data predictions.
    
        Returns:
            dict: A dictionary containing confusion matrix.
        """
        if use_train_predictions:
            y_pred = model.predict(X_train)
        else:
            y_pred = model.predict(X_test)
    
        conf_matrix = confusion_matrix(y_train if use_train_predictions else y_test, y_pred)
    
        return conf_matrix


    def find_thresholds(self, precisions, recalls, thresholds, recall_threshold=0.9, precision_threshold=0.9):
        """
        Find the thresholds for the given recall and precision thresholds.

        Parameters:
        y_test (array-like): True binary labels.
        y_probs (array-like): Target scores, can either be probability estimates of the positive class,
                            confidence values, or non-thresholded measure of decisions.
        recall_threshold (float): The desired recall threshold.
        precision_threshold (float): The desired precision threshold.

        Returns:
        tuple: (recall_threshold_value, precision_threshold_value)
            If the desired threshold is not achieved, None is returned for that threshold.
        """

        # Find the threshold for the desired recall
        recall_threshold_value = None
        try:
            recall_threshold_value = thresholds[np.where(recalls >= recall_threshold)[0][-1]]
        except IndexError:
            print(f"No threshold found that achieves {recall_threshold * 100}% recall.")

        if recall_threshold_value is not None:
            print(f"Threshold for {recall_threshold * 100}% recall: {recall_threshold_value}")

        # Find the threshold for the desired precision
        precision_threshold_value = None
        try:
            precision_threshold_value = thresholds[np.where(precisions >= precision_threshold)[0][0]]
        except IndexError:
            print(f"No threshold found that achieves {precision_threshold * 100}% precision.")

        if precision_threshold_value is not None:
            print(f"Threshold for {precision_threshold * 100}% precision: {precision_threshold_value}")

        return recall_threshold_value, precision_threshold_value