import pandas as pd
import yaml

from service.utils import Columns

PATH_CONFIG_FILE = "./service/configs/common-data.cfg.yml"

with open(PATH_CONFIG_FILE) as models_config:
    data = yaml.safe_load(models_config)

registered_model = data["registered_model"]
explained_model = data["explained_model"]
popular_items = pd.read_csv(data["popular_items"])[Columns.Item].tolist()
interactions = pd.read_csv(data["interactions"])
