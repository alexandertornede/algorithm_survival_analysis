# Code for paper: "Run2Survive: A Decision-theoretic Approach to Algorithm Selection based on Survival Analysis"

This repository holds the code for our paper "Run2Survive: A Decision-theoretic Approach to Algorithm Selection based on Survival Analysis" by Alexander Tornede, Marcel Wever, Stefan Werner, Felix Mohr and Eyke Hüllermeier. Regarding questions please contact alexander.tornede@upb.de .

## Abstract
Algorithm selection (AS) deals with the automatic selection of an algorithm from a fixed set of candidate algorithms most suitable for a specific instance of an algorithmic problem class, where “suitability” often refers to an algorithm’s runtime. Due to possibly extremely long runtimes of candidate algorithms, training data for algorithm selection models is usually generated under time constraints in the sense that not all algorithms are run to completion on all instances. Thus, training data usually comprises censored information, as the true runtime of algorithms timed out remains unknown. However, many standard AS approaches are not able to handle such information in a proper way. On the other side, survival analysis (SA) naturally supports censored data and offers appropriate ways to use such data for learning distributional models of algorithm runtime, as we demonstrate in this work. We leverage such models as a basis of a sophisticated decision-theoretic approach to algorithm selection, which we dub Run2Survive. Moreover, taking advantage of a framework of this kind, we advocate a risk-averse approach to algorithm selection, in which the avoidance of a timeout is given high priority. In an extensive experimental study with the standard benchmark ASlib, our approach is shown to be highly competitive and in many cases even superior to state-of-the-art AS approaches.

## Execution Details (Getting the Code To Run)
For the sake of reproducibility, we will detail how to reproduce the results presented in the paper below.

### 1. Configuration
In order to reproduce the results by running our code, we assume that you have a MySQL server with version >=5.7.9 running.

As a next step, you have to create a configuration file entitled `experiment_configuration.cfg` in the `conf` folder either on the top level of your IDE project next to the `allocate.txt`. This configuration file should contain the following information:

```
[DATABASE]
host = my.sqlserver.com
username = username
password = password
database = databasename
table = tablename
ssl = true

[EXPERIMENTS]
scenarios=ASP-POTASSCO,BNSL-2016,CPMP-2015,CSP-2010,CSP-MZN-2013,CSP-Minizinc-Time-2016,GRAPHS-2015,MAXSAT-PMS-2016,MAXSAT-WPMS-2016,MAXSAT12-PMS,MAXSAT15-PMS-INDU,MIP-2016,PROTEUS-2014,QBF-2011,QBF-2014,QBF-2016,SAT03-16_INDU,SAT11-HAND,SAT11-INDU,SAT11-RAND,SAT12-ALL,SAT12-HAND,SAT12-INDU,SAT12-RAND,SAT15-INDU,TSP-LION2015
approaches=sbs,oracle,per_algorithm_regressor,multiclass_algorithm_selector,satzilla-11,isac,sunny,ExpectationSurvivalForest,PAR10SurvivalForest,SurrogateAutoSurvivalForest
amount_of_training_scenario_instances=-1
amount_of_cpus=16
tune_hyperparameters=0
train_status=clip_censored
```

You have to adapt all entries below the `[DATABASE]` tag according to your database server setup. The entries have the following meaning:
* `host`: the address of your database server
* `username`: the username the code can use to access the database
* `password`: the password the code can use to access the database
* `database`: the name of the database where you imported the tables
* `table`: the name of the table, where results should be stored. This is created automatically by the code if it does not exist yet and should NOT be created manually.
* `ssl`: whether ssl should be used or not

Entries below the `[EXPERIMENTS]` define which experiments will be run. The configuration above will produce the main results presented in the paper.

### 2. Packages and Dependencies
For running the code several dependencies have to be fulfilled. The easiest way of getting there is by using [Anaconda](https://anaconda.org/). For this purpose you find an Anaconda environment definition called `survival_analysis_environment.yml` in the `singularity` folder at the top-level of this project.  Assuming that you have Anaconda installed, you can create an according environment with all required packages via

```
conda env create -f survival_analysis_environment.yml
``` 

which will create an environment named `survival_tests`. After it has been successfully installed, you can use 
```
conda activate survival_tests
```
to activate the environment and run the code (see step 4).

### 3. ASLib Data
Obviously, the code requires access to the ASLib scenarios in order to run the requested evaluations. It expects the ASLib scenarios (which can be downloaded from [Github](https://github.com/coseal/aslib_data)) to be located in a folder `data/aslib_data-master` on the top-level of your IDE project. I.e. your folder structure should look similar to this: 
```
./survival_tests
./survival_tests/approaches
./survival_tests/approaches/survival_forests
./survival_tests/results
./survival_tests/singularity
./survival_tests/baselines
./survival_tests/data
./survival_tests/data/aslib_data-master
./survival_tests/conf
```


### 4. Evaluation Results
At this point you should be good to go and can execute the experiments by running the `run.py` on the top-level of the project. 

 All results will be stored in the table given in the configuration file and has the following columns:

* `scenario_name`: The name of the scenario.
* `fold`: The train/test-fold associated with the scenario which is considered for this experiment
* `approach`: The approach which achieved the reported results, where `Run2SurviveExp := Expectation_algorithm_survival_forest`, `Run2SurvivaPar10 := PAR10_algorithm_survival_forest` and `Run2SurvivePolyLog := SurrogateAutoSurvivalForest`
* `metric`: The metric which was used to generate the result. For the `number_unsolved_instances` metric, the suffix `True` indicates that feature costs are accounted for whereas for `False` this is not the case. All other metrics automatically incorporate feature costs.
* `result`: The output of the corresponding metric.

### 5. Baseline Adaptations
Remember that for the baselines different ways of dealing with the censored information are considered. When using the configuration given above, you will obtain the results of the different Run2Survive variants and all baselines where censored information is treated according to the "runtime" strategy mentioned in the paper. For obtaining the baseline results associated with the PAR10 strategy, you have to exchange the `[EXPERIMENTS]` part of the configuration by the following snippet

```
[EXPERIMENTS]
scenarios=ASP-POTASSCO,BNSL-2016,CPMP-2015,CSP-2010,CSP-MZN-2013,CSP-Minizinc-Time-2016,GRAPHS-2015,MAXSAT-PMS-2016,MAXSAT-WPMS-2016,MAXSAT12-PMS,MAXSAT15-PMS-INDU,MIP-2016,PROTEUS-2014,QBF-2011,QBF-2014,QBF-2016,SAT03-16_INDU,SAT11-HAND,SAT11-INDU,SAT11-RAND,SAT12-ALL,SAT12-HAND,SAT12-INDU,SAT12-RAND,SAT15-INDU,TSP-LION2015
approaches=sbs,oracle,per_algorithm_regressor,multiclass_algorithm_selector,satzilla-11,isac,sunny
amount_of_training_scenario_instances=-1
amount_of_cpus=16
tune_hyperparameters=0
train_status=all
```

To obtain the baselines results associated with the "ignored" strategy, the following snippet has to be used
```
[EXPERIMENTS]
scenarios=ASP-POTASSCO,BNSL-2016,CPMP-2015,CSP-2010,CSP-MZN-2013,CSP-Minizinc-Time-2016,GRAPHS-2015,MAXSAT-PMS-2016,MAXSAT-WPMS-2016,MAXSAT12-PMS,MAXSAT15-PMS-INDU,MIP-2016,PROTEUS-2014,QBF-2011,QBF-2014,QBF-2016,SAT03-16_INDU,SAT11-HAND,SAT11-INDU,SAT11-RAND,SAT12-ALL,SAT12-HAND,SAT12-INDU,SAT12-RAND,SAT15-INDU,TSP-LION2015
approaches=sbs,oracle,per_algorithm_regressor,multiclass_algorithm_selector,satzilla-11,isac,sunny
amount_of_training_scenario_instances=-1
amount_of_cpus=16
tune_hyperparameters=0
train_status=ignore_censored
```

### 6. Generating Plots
All plots found in the paper can be generated using the self-explanatory Jupyter notebook `visualization.ipynb` in the top-level `results` folder.
