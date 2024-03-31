from mloptimizer.aux.utils import create_optimization_folder, init_logger
import os
import shutil
from datetime import datetime
import importlib
import joblib
import pandas as pd


class Tracker:
    """
    Tracker class for logging and tracking the optimization process.

    Parameters
    ----------
    name : str
        Name of the optimization process.
    folder : str
        Folder where the optimization process will be stored.
    log_file : str
        Name of the log file.
    use_mlflow : bool
        If True, the optimization process will be tracked using MLFlow.
    """

    def __init__(self, name, folder=os.curdir, log_file="mloptimizer.log", use_mlflow=False):

        self.name = name
        self.gen = 0
        # Main folder, current by default
        self.folder = create_optimization_folder(folder)
        # Log files
        self.mloptimizer_logger, self.log_file = init_logger(log_file, self.folder)
        self.optimization_logger = None

        # Paths
        self.opt_run_folder = None
        self.opt_run_checkpoint_path = None
        self.progress_path = None
        self.progress_path = None
        self.results_path = None
        self.graphics_path = None

        # MLFlow
        self.use_mlflow = use_mlflow

        if self.use_mlflow:
            self.mlflow = importlib.import_module("mlflow")

    def start_optimization(self, opt_class):
        """
        Start the optimization process.

        Parameters
        ----------
        opt_class : str
            Name of the optimization class.
        """
        # Inform the user that the optimization is starting
        self.mloptimizer_logger.info(f"Initiating genetic optimization...")
        # self.mloptimizer_logger.info("Algorithm: {}".format(type(self).__name__))
        self.mloptimizer_logger.info(f"Algorithm: {opt_class}")

    def start_checkpoint(self, opt_run_folder_name):
        """
        Start a checkpoint for the optimization process.

        Parameters
        ----------
        opt_run_folder_name : str
            Name of the folder where the checkpoint will be stored. (not the full path)
        """
        # Create checkpoint_path from date and algorithm
        if not opt_run_folder_name:
            opt_run_folder_name = "{}_{}".format(
                datetime.now().strftime("%Y%m%d_%H%M%S"),
                type(self).__name__)

        if self.use_mlflow:
            self.mlflow.set_experiment(opt_run_folder_name)

        self.opt_run_folder = os.path.join(self.folder, opt_run_folder_name)
        self.opt_run_checkpoint_path = os.path.join(self.opt_run_folder, "checkpoints")
        self.results_path = os.path.join(self.opt_run_folder, "results")
        self.graphics_path = os.path.join(self.opt_run_folder, "graphics")
        self.progress_path = os.path.join(self.opt_run_folder, "progress")

        if os.path.exists(self.opt_run_folder):
            shutil.rmtree(self.opt_run_folder)
        os.mkdir(self.opt_run_folder)
        os.mkdir(self.opt_run_checkpoint_path)
        os.mkdir(self.results_path)
        os.mkdir(self.graphics_path)
        os.mkdir(self.progress_path)
        self.optimization_logger, _ = init_logger(
            os.path.join(self.opt_run_folder, "opt.log")
        )

    def log_clfs(self, classifiers_list: list, generation: int, fitness_list: list[int]):
        self.gen = generation
        for i in range(len(classifiers_list)):
            self.optimization_logger.info(f"Generation {generation} - Classifier TOP {i}")
            self.optimization_logger.info(f"Classifier: {classifiers_list[i]}")
            self.optimization_logger.info(f"Fitness: {fitness_list[i]}")
            self.optimization_logger.info("Hyperparams: {}".format(str(classifiers_list[i].get_params())))
        self.gen = generation + 1

    def log_evaluation(self, classifier, metrics):
        self.optimization_logger.info(f"Adding to mlflow...\nClassifier: {classifier}\nMetrics: {metrics}")

        if self.use_mlflow:
            with self.mlflow.start_run():
                self.mlflow.log_params(classifier.get_params())
                # We use the generation as the step
                # self.mlflow.log_metric(key="fitness", value=metric, step=self.gen)
                self.mlflow.log_metrics(metrics, step=self.gen)

    def load_checkpoint(self, checkpoint):

        # Extract checkpoint_path from checkpoint file
        self.opt_run_checkpoint_path = os.path.dirname(checkpoint)
        self.opt_run_folder = os.path.dirname(self.opt_run_checkpoint_path)
        self.optimization_logger, _ = init_logger(os.path.join(self.opt_run_folder,
                                                               f"opt_{os.path.basename(checkpoint)}.log"))
        self.optimization_logger.info("Initiating from checkpoint {}...".format(checkpoint))

        self.results_path = os.path.join(self.opt_run_folder, "results")
        self.graphics_path = os.path.join(self.opt_run_folder, "graphics")
        self.progress_path = os.path.join(self.opt_run_folder, "progress")
        cp = joblib.load(checkpoint)
        return cp

    def write_logbook_file(self, logbook, filename=None):
        """
        Method to write the logbook to a csv file

        Parameters
        ----------
        logbook : ~deap.tools.Logbook
            logbook of the optimization process
        filename : str, optional (default=None)
            filename to save the logbook
        """
        if filename is None:
            filename = os.path.join(self.results_path, 'logbook.csv')
        pd.DataFrame(logbook).to_csv(filename, index=False)

    def write_population_file(self, populations, filename=None):
        """
        Method to write the population to a csv file

        Parameters
        ----------
        filename : str, optional (default=None)
            filename to save the population
        """
        if filename is None:
            filename = os.path.join(self.results_path, 'populations.csv')
        populations.sort_values(by=['fitness'], ascending=False
                                ).to_csv(filename, index=False)

    def start_progress_file(self, gen: int):
        progress_gen_path = os.path.join(self.progress_path, "Generation_{}.csv".format(gen))
        header_progress_gen_file = "i;total;Individual;fitness\n"
        with open(progress_gen_path, "w") as progress_gen_file:
            progress_gen_file.write(header_progress_gen_file)
            progress_gen_file.close()
        self.optimization_logger.info("Generation: {}".format(gen))

    def append_progress_file(self, gen, c, evaluations_pending, ind_formatted, fit):
        self.optimization_logger.info(
            "Fitting individual (informational purpose): gen {} - ind {} of {}".format(
                gen, c, evaluations_pending
            )
        )
        progress_gen_path = os.path.join(self.progress_path, "Generation_{}.csv".format(gen))
        with open(progress_gen_path, "a") as progress_gen_file:
            progress_gen_file.write(
                "{};{};{};{}\n".format(c,
                                       evaluations_pending,
                                       ind_formatted, fit)
            )
