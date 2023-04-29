import pandas as pd
import yaml


PATH_CONFIG_FILE = "./service/configs/common-data.cfg.yml"

with open(PATH_CONFIG_FILE) as models_config:
    data = yaml.safe_load(models_config)

registered_model = data["registered_model"]
explained_model = data["explained_model"]
popular_items = pd.read_csv(data["popular_items"])["item_id"].tolist()

interactions = pd.read_csv(data["interactions"])
users_inv_mapping = dict(enumerate(interactions["user_id"].unique()))
users_mapping = {v: k for k, v in users_inv_mapping.items()}
items_inv_mapping = dict(enumerate(interactions["item_id"].unique()))
items_mapping = {v: k for k, v in items_inv_mapping.items()}
