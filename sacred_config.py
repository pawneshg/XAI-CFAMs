import os
from sacred import Experiment

EXPERIMENT_CONFIG = os.getenv("CONFIG_FILE", "./server_config.json")
EXPERIMENT_LOCAL_CONFIG = "./local_config.json"
EXPERIMENT_NAME = "Class Activation Maps"
ex = Experiment(EXPERIMENT_NAME + "__" + EXPERIMENT_CONFIG)


@ex.named_config
def cam_server_config():
    ex.add_config(EXPERIMENT_CONFIG)


@ex.named_config
def cam_local_config():
    ex.add_config(EXPERIMENT_LOCAL_CONFIG)
