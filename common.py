import os, yaml
module_root_directory_randoms = os.path.dirname(os.path.realpath(__file__))
config_file_randoms = os.path.join(module_root_directory_randoms,'configuration.yaml')

def get_config_randoms(item, exception_if_missing_or_empty=False):
    if not os.path.exists(config_file_randoms):
        raise Exception(f"Configuration file not found at {config_file_randoms}")
    with open(config_file_randoms, 'r') as file:
        y:dict = yaml.safe_load(file)
        value = y.get(item,None)
        if value is None and exception_if_missing_or_empty:
            raise Exception(f"{item} in {config_file_randoms} was missing or empty and is required")
        return value
    