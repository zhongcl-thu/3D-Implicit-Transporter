from .transporter_net import *
import ipdb

def model_entry(config, public_params):
    config_new = dict(**config["kwargs"], **public_params)
    return globals()[config["type"]](**config_new)