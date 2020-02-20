from aslib_scenario.aslib_scenario import ASlibScenario
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import logging
#import matplotlib.pyplot as plt
#import seaborn as sns


class HierarchicalExpectedTimePerAlgorithmSurvivalForest:

    def __init__(self):
        self.logger = logging.getLogger("expected_time_per_algorithm_survival_forest")
        self.logger.addHandler(logging.StreamHandler())
        self.trained_survival_models = list()
        self.trained_classification_models = list()
        self.trained_imputers = list()
        self.trained_scalers = list()
        self.num_algorithms = 0
        self.algorithm_cutoff_time = -1;

        self.n_estimators = 100 #1000
        self.min_samples_split = 10
        self.min_samples_leaf = 15
        self.min_weight_fraction_leaf = 0.0
        self.max_features = "sqrt"
        self.bootstrap = True
        self.oob_score = False

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        print("Run fit on " + self.get_name() + " for fold " + str(fold))
        self.num_algorithms = len(scenario.algorithms)
        self.algorithm_cutoff_time = scenario.algorithm_cutoff_time

        for algorithm_id in range(self.num_algorithms):
            X_train, y_train, y_train_classification = self.get_sa_x_y(scenario, amount_of_training_instances, algorithm_id, fold)

            # impute missing values
            imputer = SimpleImputer()
            X_train = imputer.fit_transform(X_train)
            self.trained_imputers.append(imputer)

            # standardize feature values
            standard_scaler = StandardScaler()
            X_train = standard_scaler.fit_transform(X_train)
            self.trained_scalers.append(standard_scaler)

            model = RandomSurvivalForest(n_estimators=self.n_estimators,
                                         min_samples_split=self.min_samples_split,
                                         min_samples_leaf=self.min_samples_leaf,
                                         min_weight_fraction_leaf = self.min_weight_fraction_leaf,
                                         max_features=self.max_features,
                                         bootstrap = self.bootstrap,
                                         oob_score= self.oob_score,
                                         n_jobs=1,
                                         random_state=fold)
            model.fit(X_train, y_train)
            self.trained_survival_models.append(model)

            classification_model = RandomForestClassifier(random_state=fold, n_estimators=1000)
            classification_model.fit(X_train, y_train_classification)
            self.trained_classification_models.append(classification_model)

        print("Finished training " + str(self.num_algorithms) + " models on " + str(amount_of_training_instances) + " instances.")

    def predict(self, features_of_test_instance, instance_id: int):
        predicted_risk_scores = list()

        for algorithm_id in range(self.num_algorithms):
            X_test = np.reshape(features_of_test_instance, (1, len(features_of_test_instance)))

            imputer = self.trained_imputers[algorithm_id]
            X_test = imputer.transform(X_test)

            scaler = self.trained_scalers[algorithm_id]
            X_test = scaler.transform(X_test)

            model = self.trained_survival_models[algorithm_id]

            expected_survival_time = 0
            survival_function = model.predict_survival_function(X_test)[0]
            last_event_time = 0
            last_event_probability = 1
            for i in range(0, len(model.event_times_)):
                current_event_time = model.event_times_[i]
                expected_survival_time += (last_event_probability*abs(current_event_time-last_event_time))
                last_event_probability = survival_function[i]
                last_event_time = current_event_time

            #expected_survival_time += (last_event_probability*abs(self.algorithm_cutoff_time-last_event_time))
            predicted_risk_score = expected_survival_time

            classification_model = self.trained_classification_models[algorithm_id]
            classification_prediction = classification_model.predict(X_test)[0]
            #index_of_termination_class = np.argwhere(classification_model.classes_ == 1)[0][0]
            #termination_probability = classification_prediction[index_of_termination_class]
            #print(termination_probability)
            #if termination_probability < 0.65:
            if classification_prediction < 1:
                # if we do not believe that it terminates, we set its timeout super high
                predicted_risk_score = self.algorithm_cutoff_time*10

            predicted_risk_scores.append(predicted_risk_score)

        return np.asarray(predicted_risk_scores)

    def get_sa_x_y(self, scenario: ASlibScenario, num_requested_instances: int, algorithm_id: int, fold: int):
        amount_of_training_instances = min(num_requested_instances,
                                           len(scenario.instances)) if num_requested_instances > 0 else len(
            scenario.instances)
        resampled_scenario_feature_data, resampled_scenario_performances = resample(scenario.feature_data,
                                                                                    scenario.performance_data,
                                                                                    n_samples=amount_of_training_instances,
                                                                                    random_state=fold)  # scenario.feature_data, scenario.performance_data #

        X_for_algorithm_id, y_for_algorithm_id, classification_y_for_algorithm_id = self.construct_sa_dataset_for_algorithm_id(resampled_scenario_feature_data,
                                                                                            resampled_scenario_performances, algorithm_id,
                                                                                            scenario.algorithm_cutoff_time)

        return X_for_algorithm_id, y_for_algorithm_id, classification_y_for_algorithm_id


    def construct_sa_dataset_for_algorithm_id(self, instance_features, performances, algorithm_id: int,
                                              algorithm_cutoff_time):
        performances_of_algorithm_with_id = performances.iloc[:, algorithm_id].to_numpy() if isinstance(performances, pd.DataFrame) else performances[:, algorithm_id]
        num_instances = len(performances_of_algorithm_with_id)
        finished_before_timeout = np.empty(num_instances, dtype=bool)
        finished_before_timeout_classification = np.empty(num_instances, dtype=int)
        for i in range(0, len(performances_of_algorithm_with_id)):
            finished_before_timeout[i] = True if (
                    performances_of_algorithm_with_id[i] < algorithm_cutoff_time) else False
            finished_before_timeout_classification[i] = 1 if (
                    performances_of_algorithm_with_id[i] < algorithm_cutoff_time) else 0

            if performances_of_algorithm_with_id[i] >= algorithm_cutoff_time:
                performances_of_algorithm_with_id[i] = (algorithm_cutoff_time * 10)

        status_and_performance_of_algorithm_with_id = np.empty(dtype=[('cens', np.bool), ('time', np.float)],
                                                               shape=instance_features.shape[0])
        status_and_performance_of_algorithm_with_id['cens'] = finished_before_timeout
        status_and_performance_of_algorithm_with_id['time'] = performances_of_algorithm_with_id

        if isinstance(instance_features, pd.DataFrame):
            instance_features = instance_features.to_numpy()

        return instance_features, status_and_performance_of_algorithm_with_id.T, finished_before_timeout_classification

    def get_name(self):
        return "hierarchical_expected_time_per_algorithm_survival_forest"


    def set_parameters(self, parametrization):
        self.n_estimators = parametrization["n_estimators"]
        self.min_samples_split = parametrization["min_samples_split"]
        self.min_samples_leaf = parametrization["min_samples_leaf"]
        self.min_weight_fraction_leaf = parametrization["min_weight_fraction_leaf"]
        self.max_features = parametrization["max_features"]
        self.bootstrap = parametrization["bootstrap"]
        self.oob_score = parametrization["oob_score"]
