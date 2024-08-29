import yaml
import logging
import sys

from typing import Dict, List

def read_yaml_file(file_path : str) -> Dict:
    """Utility function to read a yaml file.

    Args:
        file_path (str): Path to the yaml file to read.

    Returns:
        dict: The contents of the yaml file as a dictionary.
    """
    logging.info(f"Reading yaml file: {file_path}")
    try:
        with open(file_path, "r", ) as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        sys.exit(1)

def check_required_keys(dict : dict, required_keys : List[str]):
    """Utility function to check if a dictionary contains all required keys.

    Args:
        dict (dict): Dictionary to check.
        required_keys (list): List of required keys.
    """
    logging.debug(f"Checking Dictionary {dict} for required keys {required_keys}")
    for key in required_keys:
        if key not in dict:
            logging.error(f"Missing required key: {key}")
            sys.exit(1)