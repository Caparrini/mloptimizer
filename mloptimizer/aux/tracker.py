from mloptimizer.utils import create_optimization_folder, init_logger
import os
import shutil
from datetime import datetime


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
    """
    def __init__(self, name, folder=os.curdir, log_file="mloptimizer.log"):
        self.name = name
        self.bugs = []
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
