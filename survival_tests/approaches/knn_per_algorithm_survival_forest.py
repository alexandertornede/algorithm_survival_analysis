from aslib_scenario.aslib_scenario import ASlibScenario
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import logging
import copy
from approaches.per_algorithm_survival_forest import PerAlgorithmSurvivalForest
#import matplotlib.pyplot as plt
#import seaborn as sns


class KnnPerAlgorithmSurvivalForest:

    def __init__(self, k: int):
        self.logger = logging.getLogger("knn_per_algorithm_survival_forest")
        self.logger.addHandler(logging.StreamHandler())
        self.num_algorithms = 0
        self.algorithm_cutoff_time = -1;
        self.X_train = None
        self.y_train = None
        self.scenario = None
        self.fold = -1
        self.k = k
        self.imputer = SimpleImputer()
        self.standard_scaler = StandardScaler()


    def get_k_closest_training_instances(self, test_instance: np.ndarray):
        distances = list()
        for i in range(0, len(self.X_train)):
            row = self.X_train[i]
            distance = np.linalg.norm(row - test_instance)
            distances.append(distance)

        sorted_indices = np.argsort(np.asarray(distances))

        X_closest_training_instances = np.empty([self.k, len(test_instance)])
        y_closest_training_instances = np.empty([self.k, self.num_algorithms])

        for i in range(0, self.k):
            instance_index = sorted_indices[i]
            X_closest_training_instances[i] = self.X_train[instance_index]
            y_closest_training_instances[i] = self.y_train[instance_index]

        return X_closest_training_instances, y_closest_training_instances


    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        self.num_algorithms = len(scenario.algorithms)
        self.algorithm_cutoff_time = scenario.algorithm_cutoff_time
        self.scenario = scenario
        self.fold = fold

        X_train = self.scenario.feature_data.to_numpy()
        X_train = self.imputer.fit_transform(X_train)
        self.X_train = self.standard_scaler.fit_transform(X_train)

        self.y_train = self.scenario.performance_data.to_numpy()


    def predict(self, features_of_test_instance, instance_id: int):
        print("predict on knn")

        X_test = np.reshape(features_of_test_instance, (1, len(features_of_test_instance)))
        X_test = self.imputer.transform(X_test)
        X_test = self.standard_scaler.transform(X_test)[0]


        X_k_train, y_k_train = self.get_k_closest_training_instances(X_test)

        adapted_scenario = copy.deepcopy(self.scenario)
        adapted_scenario.feature_data = X_k_train
        adapted_scenario.performance_data = y_k_train

        approach = PerAlgorithmSurvivalForest()
        approach.fit(adapted_scenario, self.fold, self.k)

        predicted_risk_scores = approach.predict(X_test, instance_id)

        return predicted_risk_scores

    def get_name(self):
        return "knn_per_algorithm_survival_forest"
