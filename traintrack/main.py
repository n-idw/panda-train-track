import argparse
import logging
import os

from traintrack.utils.io import read_yaml_file
from traintrack.stage    import stage

def main():
    """This is the main function that is executed when using traintrack as a command line utility. It parses the command line arguments and runs the pipeline stages.
    """

    # Set the number of threads for NumExpr to the number of available CPUs.
    os.environ["NUMEXPR_MAX_THREADS"] = str(os.cpu_count())

    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description="Utility to access and configure different parts of the machine learning pipeline.")

    parser.add_argument("pipeline_config_file", type=str, help="Path to the .yaml file containing the pipeline configuration.")
    
    parser.add_argument("-i", "--inference", action="store_true", help="Run the pipeline stages in inference mode.")
    parser.add_argument("-v", "--verbose"  , action="count"     , help="Increases the verbosity: -v = INFO, -vv = DEBUG.", default=0)

    command_line_args = parser.parse_args()

    # Set the logging level based on the verbosity.
    if command_line_args.verbose == 1:
        logging.basicConfig(level=logging.INFO,    format=" %(levelname)s in %(filename)s :: %(message)s")
    elif command_line_args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG,   format=" %(levelname)s in %(pathname)s :: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format=" %(levelname)s in %(filename)s :: %(message)s")        

    # Log the parsed command line arguments.
    if command_line_args.verbose > 0:
        logging.info("Parsed command line arguments:")
        for pipeline_arg in vars(command_line_args):
            logging.info(f"{pipeline_arg}: {getattr(command_line_args, pipeline_arg)}")

    # Read the pipeline configuration file and extract the stage list containing the stage dependent configuration.
    pipeline_config = read_yaml_file(command_line_args.pipeline_config_file)
    stage_list = pipeline_config["stage_list"]
    del pipeline_config["stage_list"]

    # Run the pipeline stages.
    for stage_config in stage_list:
        stage(stage_config, pipeline_config, command_line_args.inference).run()

# Run the main function if the script is executed.
if __name__ == "__main__":
    main()