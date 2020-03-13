import logging
import numpy as np
from itertools import chain, combinations
from aslib_scenario.aslib_scenario import ASlibScenario
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.neighbors import KDTree


class SUNNY:
    def __init__(self):
        self._name = 'SUNNY'
        self._imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self._scaler = StandardScaler()
        self._k = 16

    def get_name(self):
        return self._name

    def fit(self, scenario: ASlibScenario, fold: int, num_instances: int):
        # TODO: assert assumptions, e.g. presolver, ...?
        self._num_algorithms = len(scenario.algorithms)
        self._algorithm_cutoff_time = scenario.algorithm_cutoff_time

        # resample `amount_of_training_instances` instances and preprocess them accordingly
        num_instances = min(num_instances, len(
            scenario.instances)) if num_instances > 0 else len(scenario.instances)
        features, performances = self._resample_instances(
            scenario.feature_data, scenario.performance_data, num_instances, random_state=fold)
        features, performances = self._preprocess_scenario(
            scenario, features, performances)

        # build nearest neighbors index
        self._model = KDTree(features, leaf_size=30, metric='euclidean')
        self._performances = np.copy(performances)

    def predict(self, features, instance_id: int):
        assert(features.ndim == 1), '`features` must be one dimensional'
        features = np.expand_dims(features, axis=0)
        features = self._imputer.transform(features)
        features = self._scaler.transform(features)

        neighbour_idx = np.squeeze(self._model.query(
            features, k=self._k, return_distance=False))
        sub_portfolio = self._build_subportfolio(neighbour_idx)
        schedule = self._build_schedule(neighbour_idx, sub_portfolio)

        # in this setting, solely return the algorithm scheduled first by SUNNY
        return schedule[0]

    def _build_subportfolio(self, neighbour_idx):
        sub_performances = self._performances[neighbour_idx, :]

        # TODO: naive computation, inefficient / infeasible for > 10 - 15 algorithms
        algorithms = range(self._num_algorithms)
        num_solved, avg_time = np.NINF, np.NINF
        sub_portfolio = None
        for subset in chain.from_iterable(combinations(algorithms, n) for n in range(1, len(algorithms))):
            # compute number of solved instances and average solving time
            tmp_solved = np.count_nonzero(
                np.min(sub_performances[:, subset], axis=1) < self._algorithm_cutoff_time)
            # TODO: not entirely sure whether this is the correct way to compute the average runtime
            tmp_avg_time = np.sum(
                sub_performances[:, subset]) / sub_performances[:, subset].size
            if tmp_solved > num_solved or (tmp_solved == num_solved and tmp_avg_time < avg_time):
                num_solved, avg_time = tmp_solved, tmp_avg_time
                sub_portfolio = subset

        return sub_portfolio

    def _build_schedule(self, neighbour_idx, sub_portfolio):
        # schedule algorithms wrt. to solved instances (asc.) and break ties according to its average runtime (desc.)
        sub_performances = self._performances[neighbour_idx, :]
        alg_performances = {alg: (np.count_nonzero(
            sub_performances[:, alg] < self._algorithm_cutoff_time), (-1)*np.sum(sub_performances[:, alg])) for alg in sub_portfolio}
        schedule = sorted([(solved, avg_time, alg) for (
            alg, (solved, avg_time)) in alg_performances.items()], reverse=True)

        return [alg for (_, _, alg) in schedule]

    def _resample_instances(self, feature_data, performance_data, num_instances, random_state):
        return resample(feature_data, performance_data, n_samples=num_instances, random_state=random_state)

    def _preprocess_scenario(self, scenario, features, performances):
        # TODO: what to do about missing features? -> generally there should not be any missing features,
        # since, in this case, the backup solver is selected
        features = self._imputer.fit_transform(features)
        features = self._scaler.fit_transform(features)

        return features, performances
