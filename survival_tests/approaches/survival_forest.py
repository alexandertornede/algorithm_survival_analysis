from aslib_scenario.aslib_scenario import ASlibScenario
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import logging


class AlgorithmSurvivalForest:

    def __init__(self):
        self.logger = logging.getLogger("survival_forest")
        self.logger.addHandler(logging.StreamHandler())
        self.imputer = SimpleImputer()
        self.standard_scaler = StandardScaler()
        self.rsf = None
        self.num_algorithms = 0

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        self.num_algorithms = len(scenario.algorithms)

        X_train, y_train = self.get_x_y(scenario, amount_of_training_instances, fold)
        # impute missing values
        X_train = self.imputer.fit_transform(X_train)
        # standardize feature values
        X_train = self.standard_scaler.fit_transform(X_train)
        self.rsf = RandomSurvivalForest(n_estimators=1000,
                                   min_samples_split=10,
                                   min_samples_leaf=15,
                                   max_features="sqrt",
                                   n_jobs=1,
                                   random_state=fold)
        self.logger.info("Training on " + str(len(X_train)) + " survival instances with random state " + str(self.rsf.random_state))
        self.rsf.fit(X_train, y_train)
        self.logger.info("Finshed training.")


    def predict(self, features_of_test_instance, instance_id: int):
        # create X with one line per algorithm and return the resulting vector
        X_test = self.construct_test_dataset_per_instance(features_of_test_instance, self.num_algorithms)
        X_test = self.imputer.transform(X_test)
        X_test = self.standard_scaler.transform(X_test)
        return np.negative(self.rsf.predict(X_test))


    def get_x_y(self, scenario: ASlibScenario, num_requested_instances: int, fold: int):
        num_algorithms = len(scenario.algorithms)
        X = None
        y = None
        amount_of_training_instances = min(num_requested_instances,
                                           len(scenario.instances)) if num_requested_instances > 0 else len(
            scenario.instances)
        resampled_scenario_feature_data, resampled_scenario_performances = resample(scenario.feature_data,
                                                                                    scenario.performance_data,
                                                                                    n_samples=amount_of_training_instances,
                                                                                    random_state=fold)  # scenario.feature_data, scenario.performance_data #

        for i in range(0, num_algorithms):
            X_for_algorithm_id, y_for_algorithm_id = self.construct_dataset_for_algorithm_id(resampled_scenario_feature_data,
                                                                                        resampled_scenario_performances, i,
                                                                                        scenario.algorithm_cutoff_time,
                                                                                        len(scenario.algorithms))
            X = X_for_algorithm_id if X is None else np.vstack((X, X_for_algorithm_id))
            y = y_for_algorithm_id if y is None else np.append(y, y_for_algorithm_id)
        return X, y


    def construct_dataset_for_algorithm_id(self, instance_features: pd.DataFrame, performances: pd.DataFrame, algorithm_id: int,
                                           algorithm_cutoff_time, num_algorithms: int):
        performances_of_algorithm_with_id = performances.iloc[:, algorithm_id].to_numpy()
        num_instances = len(performances_of_algorithm_with_id)
        finished_before_timeout = np.empty(num_instances, dtype=bool)
        for i in range(0, len(performances_of_algorithm_with_id)):
            finished_before_timeout[i] = True if (
                    performances_of_algorithm_with_id[i] < algorithm_cutoff_time) else False

        status_and_performance_of_algorithm_with_id = np.empty(dtype=[('cens', np.bool), ('time', np.float)],
                                                               shape=instance_features.shape[0])
        status_and_performance_of_algorithm_with_id['cens'] = finished_before_timeout
        status_and_performance_of_algorithm_with_id['time'] = performances_of_algorithm_with_id
        # print(status_and_performance_of_algorithm_with_id)

        one_hot_encoding_of_algorithm = np.zeros(num_algorithms)
        one_hot_encoding_of_algorithm[algorithm_id] = 1
        algorithm_features = np.repeat([one_hot_encoding_of_algorithm], num_instances, axis=0)
        # print(algorithm_features)
        # print(np.shape(algorithm_features))
        instance_features = instance_features.to_numpy()

        instance_and_algorithm_features = np.append(instance_features, algorithm_features, axis=1)
        # print(instance_and_algorithm_features)

        return instance_and_algorithm_features, status_and_performance_of_algorithm_with_id.T

    def construct_X_for_single_algorithm(self, instance_features,
                                               num_algorithms: int, algorithm_id: int):

        one_hot_encoding_of_algorithm = np.zeros(num_algorithms)
        one_hot_encoding_of_algorithm[algorithm_id] = 1
        # print(one_hot_encoding_of_algorithm)

        instance_and_algorithm_features = np.append(instance_features, one_hot_encoding_of_algorithm, axis=0)
        # print(instance_and_algorithm_features)

        return instance_and_algorithm_features


    def construct_test_dataset_per_instance(self, instance_features, num_algorithms: int):
        X = None
        for algo_id in range(0, num_algorithms):
            X_i = self.construct_X_for_single_algorithm(instance_features, num_algorithms,
                                                              algo_id)
            if X is None:
                X = X_i
            else:
                X = np.vstack((X, X_i))
        return X

    def get_name(self):
        return "global_survival_forest"
