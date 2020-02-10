from aslib_scenario.aslib_scenario import ASlibScenario
import numpy as np
import logging
import database_utils

logger = logging.getLogger("evaluation")
logger.addHandler(logging.StreamHandler())


def publish_results_to_database(db_config, scenario_name: str, fold: int, approach: str, metric_name:str, result: float):
    db_handle, table_name = database_utils.initialize_mysql_db_and_table_name_from_config(db_config)

    db_cursor = db_handle.cursor()
    sql_statement = "INSERT INTO " + table_name + " (scenario_name, fold, approach, metric, result) VALUES (%s, %s, %s, %s, %s)"
    values = (scenario_name, fold, approach, metric_name, str(result))
    db_cursor.execute(sql_statement, values)
    db_handle.commit()

    db_cursor.close()
    db_handle.close()


def evaluate(scenario: ASlibScenario, approach, metrics, amount_of_training_instances: int, fold: int, db_config):
    np.random.seed(fold)
    metric_results = evaluate_train_test_split(scenario, approach, metrics, fold, amount_of_training_instances)
    for i, result in enumerate(metric_results):
        publish_results_to_database(db_config, scenario.scenario, fold, approach.get_name(), metrics[i].get_name(), result)


def evaluate_train_test_split(scenario: ASlibScenario, approach, metrics, fold: int, amount_of_training_instances: int):
    logger.info("-----------------------------")
    logger.info("Evaluating \"" + approach.get_name() + "\" fold " + str(fold) + " training on " + str(amount_of_training_instances) + " scenario instances on scenario " + str(scenario.scenario))
    test_scenario, train_scenario = scenario.get_split(indx=fold)

    if approach.get_name() == 'oracle':
        approach.fit(test_scenario, fold, amount_of_training_instances)
    else:
        approach.fit(train_scenario, fold, amount_of_training_instances)

    approach_metric_values = np.empty(len(metrics))

    num_counted_test_values = 0
    for instance_id in range(0, len(test_scenario.instances)):

        print(str(instance_id) + "/" + str(len(test_scenario.instances)))

        X_test = test_scenario.feature_data.to_numpy()[instance_id]
        y_test = test_scenario.performance_data.to_numpy()[instance_id]

        accumulated_feature_time = 0
        if test_scenario.feature_cost_data is not None:
            feature_time = test_scenario.feature_cost_data.to_numpy()[instance_id]
            accumulated_feature_time = np.sum(feature_time)

        contains_non_censored_value = False
        for y_element in y_test:
            if y_element < test_scenario.algorithm_cutoff_time:
                contains_non_censored_value = True
        if contains_non_censored_value:
            num_counted_test_values += 1

            predicted_scores = approach.predict(X_test, instance_id)
            for i, metric in enumerate(metrics):
                runtime = metric.evaluate(y_test, predicted_scores, accumulated_feature_time, scenario.algorithm_cutoff_time)
                approach_metric_values[i] = (approach_metric_values[i] + runtime)

    approach_metric_values = np.true_divide(approach_metric_values, num_counted_test_values)

    return approach_metric_values


def print_stats_of_scenario(scenario: ASlibScenario):
    logger.info("scenario: " + str(scenario.scenario))
    logger.info("#instances: " + str(len(scenario.instances)))
    logger.info("#features: " + str(len(scenario.feature_data.columns)))
    logger.info("#algorithms: " + str(len(scenario.algorithms)))
    logger.info("cutoff-time: " + str(scenario.algorithm_cutoff_time))


def evaluate_scenario(scenario_name: str, approach, metrics, amount_of_training_scenario_instances: int, fold: int, config):
    scenario = ASlibScenario()
    scenario.read_scenario('data/aslib_data-master/' + scenario_name)
    print_stats_of_scenario(scenario)
    evaluate(scenario, approach, metrics, amount_of_training_scenario_instances, fold, config)
    return scenario_name




