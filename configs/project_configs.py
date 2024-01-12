import os
from dataclasses import dataclass

@dataclass
class ProjectConfigs:
    # directories
    HOME_DIR: str = os.environ['HOME']
    ENCODER_CONF_PTH: str = 'encoder_configs.json'
    DECODER_CONF_PTH: str = 'decoder_configs.json'
    SCRIPT_PATH: str = './TrainingScript.py'
    INTERPRETER_PATH: str = HOME_DIR + '/anaconda3/envs/env_name/bin/python'
    QUEUE_PATH: str = './queue'
    DATA_PATH: str = '/mnt/data2/sampled_gt/random/u'
    EXPEREMENT_PATH: str = './experiments'
    DESCRIPTION_PATH: str = EXPEREMENT_PATH + 'description.json'
    SERVER_NAME: str = 'server_name'