import logging
import sys
import configparser
import multiprocessing as mp
import database_utils
from evaluation import evaluate_scenario
from approaches.single_best_solver import SingleBestSolver
from approaches.oracle import Oracle
from approaches.survival_forest import AlgorithmSurvivalForest
from approaches.per_algorithm_survival_forest import PerAlgorithmSurvivalForest
from approaches.hierarchical_expected_time_per_algorithm_survival_forest import HierarchicalExpectedTimePerAlgorithmSurvivalForest
from approaches.per_algorithm_survival_svm import PerAlgorithmSurvivalSVM
from approaches.knn_per_algorithm_survival_forest import KnnPerAlgorithmSurvivalForest
from approaches.expected_time_per_algorithm_survival_forest import ExpectedTimePerAlgorithmSurvivalForest
from approaches.survival_forests.surrogate import SurrogateSurvivalForest
from approaches.survival_forests.auto_surrogate import SurrogateAutoSurvivalForest
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.sunny import SUNNY
from baselines.snnap import SNNAP
from baselines.isac import ISAC
from baselines.satzilla11 import SATzilla11
from par_10_metric import Par10Metric
from number_unsolved_instances import NumberUnsolvedInstances
from par_10_scheduling_metric import Par10SchedulingMetric


logger = logging.getLogger("run")
logger.addHandler(logging.StreamHandler())


def initialize_logging():
    logging.basicConfig(filename='logs/log_file.log', filemode='w',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('conf/experiment_configuration.cfg'))
    return config


def print_config(config: configparser.ConfigParser):
    for section in config.sections():
        logger.info(str(section) + ": " + str(dict(config[section])))


def log_result(result):
    logger.info("Finished experiements for scenario: " + result)


def create_approach(approach_names):
    approaches = list()
    for approach_name in approach_names:
        if approach_name == 'sbs':
            approaches.append(SingleBestSolver())
        if approach_name == 'oracle':
            approaches.append(Oracle())
        if approach_name == 'survival_forest':
            approaches.append(AlgorithmSurvivalForest())
        if approach_name == 'per_algorithm_survival_forest':
            approaches.append(PerAlgorithmSurvivalForest())
        if approach_name == 'per_algorithm_survival_svm':
            approaches.append(PerAlgorithmSurvivalSVM())
        if approach_name == 'knn_per_algorithm_survival_forest':
            approaches.append(KnnPerAlgorithmSurvivalForest(20))
        if approach_name == 'ExpectationSurvivalForest':
            approaches.append(SurrogateSurvivalForest(
                criterion='Expectation'))
        if approach_name == 'PolynomialSurvivalForest':
            approaches.append(SurrogateSurvivalForest(
                criterion='Polynomial'))
        if approach_name == 'GridSearchSurvivalForest':
            approaches.append(SurrogateSurvivalForest(
                criterion='GridSearch'))
        if approach_name == 'ExponentialSurvivalForest':
            approaches.append(SurrogateSurvivalForest(
                criterion='Exponential'))
        if approach_name == 'SurrogateAutoSurvivalForest':
            approaches.append(SurrogateAutoSurvivalForest())
        if approach_name == 'hierarchical_expected_time_per_algorithm_survival_forest':
            approaches.append(
                HierarchicalExpectedTimePerAlgorithmSurvivalForest())
        if approach_name == 'per_algorithm_regressor':
            approaches.append(PerAlgorithmRegressor())
        if approach_name == 'multiclass_algorithm_selector':
            approaches.append(MultiClassAlgorithmSelector())
        if approach_name == 'sunny':
            approaches.append(SUNNY())
        if approach_name == 'snnap':
            approaches.append(SNNAP())
        if approach_name == 'satzilla-11':
            approaches.append(SATzilla11())
        if approach_name == 'isac':
            approaches.append(ISAC())
    return approaches


#######################
#         MAIN        #
#######################

initialize_logging()
config = load_configuration()
logger.info("Running experiments with config:")
print_config(config)

fold = int(sys.argv[1])
logger.info("Running experiments for fold " + str(fold))

db_handle, table_name = database_utils.initialize_mysql_db_and_table_name_from_config(
    config)
database_utils.create_table_if_not_exists(db_handle, table_name)

amount_of_cpus_to_use = int(config['EXPERIMENTS']['amount_of_cpus'])
pool = mp.Pool(amount_of_cpus_to_use)


scenarios = config["EXPERIMENTS"]["scenarios"].split(",")
approach_names = config["EXPERIMENTS"]["approaches"].split(",")
amount_of_scenario_training_instances = int(
    config["EXPERIMENTS"]["amount_of_training_scenario_instances"])
tune_hyperparameters = bool(int(config["EXPERIMENTS"]["tune_hyperparameters"]))

for fold in range(1, 11):
    for scenario in scenarios:
        approaches = create_approach(approach_names)
        print(approaches)

        if len(approaches) < 1:
            logger.error("No approaches recognized!")
        for approach in approaches:
            metrics = list()
            metrics.append(Par10Metric())
            if approach.get_name() != 'oracle':
                metrics.append(NumberUnsolvedInstances(False))
                metrics.append(NumberUnsolvedInstances(True))
                # metrics.append(Par10SchedulingMetric(2))
                # metrics.append(Par10SchedulingMetric(3))
                # metrics.append(Par10SchedulingMetric(5))
                # metrics.append(Par10SchedulingMetric(-1))
            logger.info("Submitted pool task for approach \"" +
                        str(approach.get_name()) + "\" on scenario: " + scenario)
            pool.apply_async(evaluate_scenario, args=(scenario, approach, metrics,
                                                      amount_of_scenario_training_instances, fold, config, tune_hyperparameters), callback=log_result)
            
            #evaluate_scenario(scenario, approach, metrics, amount_of_scenario_training_instances, fold, config, tune_hyperparameters)
            print('Finished evaluation of fold')

pool.close()
pool.join()
logger.info("Finished all experiments.")
