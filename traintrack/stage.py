import sys
import logging
import os
import importlib
import torch
import ast

from typing import Dict

from traintrack.utils.io import read_yaml_file, check_required_keys

from pytorch_lightning.loggers   import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer   import Trainer

class stage:
    """This class represents a stage in the machine learning pipeline. 
       It is responsible for configuring and running a specified machine learning, or data processing model using .yaml configuration files.
    """
    
    def __init__(self, stage_config : Dict, pipeline_config : Dict, inference_mode : bool):
        """Initializes the stage class with the specified configurations and checks if all required options are set.

        Args:
            stage_config (dict): Dictionary containing the stage configuration.
            pipeline_config (dict): Dictionary containing the pipeline configuration.
            inference_mode (bool): Flag to indicate if the stage is running in inference mode.
        """
        
        logging.info("Initializing stage")
        
        # Check if all required keys are present in the configuration files
        check_required_keys(dict          = pipeline_config, 
                            required_keys = ["model_library", "artifact_library", "project"])
        
        check_required_keys(dict = stage_config, 
                            required_keys = ["set", "name"])
        
        # Set the configuration files as class attributes
        self.pipeline_config = pipeline_config
        self.stage_config    = stage_config
        self.inference_mode  = inference_mode
        
        # Initialize other class attributes
        self.model_config       = None
        self.model              = None
        self.can_be_trained     = None
        self.can_process_data   = None
        self.trainer            = None


    def __get_model_class(self, model_class_name : str) -> object:
        """
        Search and return the model class

        Searches all files in the model library for the specified model class name and returns the class if found.

        Args:
            model_class_name (str): Name of the model class to search for.

        Returns:
            type: The class of the model if found.
        """

        model_dir = os.path.join(self.pipeline_config["model_library"], self.stage_config["set"], "Models")

        logging.info(f"Searching for model {model_class_name} in {model_dir}")

        found_module = False

        # Search for the model class in the python files in the specified model directory
        for root, dirs , files in os.walk(model_dir):
            logging.debug(f"Searching in {root}")
            logging.debug(f"Found files: {files}")
            logging.debug(f"Found directories: {dirs}")
            for file in files:
                # Get the full path to files with the .py extension
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    logging.debug(f"Checking file {full_path}")    
                    
                    # Read the python file to get the program tree (structure)
                    with open(full_path, 'r') as f:
                        content = f.read()
                    try:
                        tree = ast.parse(content)
                    except SyntaxError:
                        continue  # Skip files that fail to parse
                    
                    # Check if one of the class definitions in the program tree matches the specified class name
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and node.name == model_class_name:
                            # Get the module name as the relative path from the model library to the model file and then leave the for loop
                            module_name = ".".join([self.stage_config["set"], "Models", os.path.splitext(os.path.basename(full_path))[0]])
                            logging.info(f"Found model class {model_class_name} in {module_name}")
                            found_module = True
                            break
                if found_module:
                    break
                else:
                    logging.debug(f"Class not found in {file}")
            if found_module:
                break
            else:
                logging.debug(f"Class not found in {root}")
        
        if not found_module:
            logging.error(f"Model class {model_class_name} not found")
            sys.exit(1)

        # Import the module and the model class
        module = importlib.import_module(module_name)
        
        return getattr(module, model_class_name)


    def __prepare_stage(self):
        """This function imports the class specified in the stage configuration and initializes the model with the specified model configuration from the configuration yaml file.
           In the case of a machine learning model, it also initializes the PyTorch Lightning Trainer and Logger.
        """
        
        logging.info("Preparing stage")

        # Add the model library to the system path
        sys.path.append(self.pipeline_config["model_library"])

        model_class = self.__get_model_class(self.stage_config["name"])
        
        # Check if the model class contains the required methods for data processing or machine learning
        # TODO: Maybe make this more flexible in the future to include models that have slightly different method names
        self.can_be_trained   = callable(getattr(model_class, "training_step", None))
        self.can_process_data = callable(getattr(model_class, "prepare_data", None))

        # Depending on the model type, initialize the model with the specified configuration
        if self.can_be_trained:
            self.__init_ml(model_class)
        elif self.can_process_data:
            self.__init_data_processing(model_class)
        else:
            logging.error("Model must contain either a \"training_step\" or a \"prepare_data\" method")
            sys.exit(1)

    def __init_ml(self, model_class : object):
        """Function to initialize a machine learning model, logger, and trainer with the specified configuration.

        Args:
            model_class (type): The class of the machine learning model to initialize.
        """
        
        if self.inference_mode:
            logging.info("Initializing inference mode")
            if "resume_id" not in self.stage_config.keys():
                logging.error("Inference mode requires a resume id")
                sys.exit(1)
        else:
            logging.info("Initializing training mode")

        # Check if a resume id is given in the stage configuration.
        if "resume_id" in self.stage_config.keys():
            # Set the path to the checkpoint file
            checkpoint_path = os.path.join(self.pipeline_config["artifact_library"], self.pipeline_config["project"], self.stage_config["resume_id"], "checkpoints", "last.ckpt")
            try:
                with open(checkpoint_path, "r"):
                    logging.info(f"Resuming training from checkpoint {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
                    self.model_config = checkpoint["hyper_parameters"]
                    self.model_config["checkpoint_path"] = checkpoint_path
            except FileNotFoundError:
                logging.error(f"Checkpoint file not found: {checkpoint_path}")
                sys.exit(1)
        else:
            # Read the model configuration from the specified configuration file
            model_config_path = os.path.join(self.pipeline_config["model_library"], self.stage_config["set"], "configs", self.stage_config["config"])
            self.model_config = read_yaml_file(model_config_path)
            self.model_config["resume_id"] = None

        # Check if all required keys are present in the model configuration
        check_required_keys(dict          = self.model_config,
                            required_keys = ["max_epochs","callbacks"])

        # Add the command line, stage and pipeline configuration to the model configuration so that they are saved in the checkpoint
        self.model_config.update(self.stage_config)
        self.model_config.update(self.pipeline_config)
        self.model_config["inference"] = self.inference_mode
        logging.debug(self.model_config)
        
        # Initialize the machine learning model with the specified model configuration
        self.model = model_class(self.model_config)

        # Initialize the PyTorch Logger as specified in the model configuration
        if "logger" in  self.model_config.keys():
            if self.model_config["logger"] == "wandb":
                logging.info("Using the Weights and Biases logger")
                logger = WandbLogger(
                    project  = self.pipeline_config["project"],
                    save_dir = self.model_config["artifact_library"],
                    id       = self.model_config["resume_id"],
                )
            elif self.model_config["logger"] == "tb":
                logging.info("Using the TensorBoard logger")
                logger = TensorBoardLogger(
                    name     = self.pipeline_config["project"],
                    save_dir = self.model_config["artifact_library"],
                    version  = self.model_config["resume_id"],
                )
            else:
                logging.error(f"Logger {self.model_config['logger']} not recognized")
                sys.exit(1)
        else:
            logging.info("Running without a logger")

        # Configure how/when PyTorch should save the model checkpoints
        logging.info("Configuring model checkpoint callback")
        checkpoint_callback = ModelCheckpoint(
            # Use a figure of merit or the value of the loss function as the monitor learning progression of the model
            monitor    = self.model_config["fom"]        if "fom"        in self.model_config.keys() else "val_loss",
            # Save the best k checkpoints according to the monitored value
            save_top_k = self.model_config["save_top_k"] if "save_top_k" in self.model_config.keys() else 2, 
            # Decide weather to create a checkpoint if a new minimal "min" or maximal "max" value is reached
            mode       = self.model_config["fom_mode"]   if "fom_mode"   in self.model_config.keys() else "min",
            # Always save the epoch
            save_last  = True
        )

        # Create a list with all callback objects specified in the model configuration
        logging.info("Configuring callback objects")
        callback_list = [checkpoint_callback]
        for callback in self.model_config["callbacks"]:
            callback_object = self.__get_model_class(callback)()
            callback_list.append(callback_object)

        # Initialize the PyTorch Lightning Trainer
        logging.info("Configuring the trainer")
        self.trainer    = Trainer(
            max_epochs  = self.model_config["max_epochs"],
            accelerator = "gpu" if torch.cuda.is_available() else "cpu",
            devices     = 1     if torch.cuda.is_available() else os.cpu_count(),
            logger      = logger,
            callbacks   = callback_list,
        )

        # Add sanity steps to the trainer if resuming from a checkpoint
        if self.model_config["resume_id"] is not None:
            self.trainer.num_sanity_val_steps = self.model_config["sanity_steps"] if "sanity_steps" in self.model_config else 2


    def __init_data_processing(self, model_class : object):
        """Function to initialize a data processing with the specified configuration.

        Args:
            model_class (type): The class of the data processing model to initialize.
        """

        logging.info("Initializing data processing")

        # Read the model configuration from the specified configuration file
        model_config_path = os.path.join(self.pipeline_config["model_library"], self.stage_config["set"], "configs", self.stage_config["config"])
        self.model_config = read_yaml_file(model_config_path)
        logging.debug(self.model_config)

        # Initialize the processing model with the specified model configuration
        self.model = model_class(self.model_config)


    def run(self):
        """This function prepares, initializes, and runs the stage using the specified configurations.
        """

        # Prepare the given model with the specified configuration
        self.__prepare_stage()

        logging.info("Running stage")
        if self.can_be_trained:
            if self.inference_mode:
                # Load parameters of a trained model at a given checkpoint (inference mode)
                logging.info("Running inference")
                self.model.load_state_dict(torch.load(self.model_config["checkpoint_path"])["state_dict"])
            elif self.model_config["resume_id"] is not None:
                # Fit the machine learning model from the checkpoint using the initialized trainer
                logging.info("Resuming training")
                self.trainer.fit(self.model, ckpt_path=self.model_config["checkpoint_path"])
            else:
                # Fit the machine learning model from the beginning using the initialized trainer
                logging.info("Running training")
                self.trainer.fit(self.model)
            # Evaluate the trained model on the test dataset
            self.trainer.test(self.model)
        elif self.can_process_data:
            # Process the data using the initialized data processing model
            logging.info("Running data processing")
            self.model.prepare_data()
