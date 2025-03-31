import json
import os.path as osp
import os
import glob
import math
import numpy as np
import torch
import argparse
import yaml
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, default="configs/config.yaml")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)
cfg = config["regression"]

DATA_ROOT       = cfg["directories"]["data"]["data_root"]
PARA_SUBDIR     = cfg["directories"]["data"]["para_subdir"]
NORM_SUBDIR     = cfg["directories"]["data"]["norm_subdir"]
NORMALIZE       = cfg["preprocess"]["scaler"]
UNNORMALIZE     = cfg["preprocess"]["descaler"]

para_paths = sorted(glob.glob(osp.join(DATA_ROOT, PARA_SUBDIR, "*.json")))
norm_path = osp.join(DATA_ROOT, NORM_SUBDIR)
os.makedirs(norm_path, exist_ok=True)

utils = importlib.import_module("utils")
scaler = getattr(utils, NORMALIZE)
descaler = getattr(utils, UNNORMALIZE)

dynVisc = []
kinVisc = [] 
surfT = []
density = []

for path in para_paths:
    with open(path, 'r') as file:
        data = json.load(file)
        dynVisc.append(data["dynamic_viscosity"])
        kinVisc.append(data["kinematic_viscosity"])
        surfT.append(data["surface_tension"])
        density.append(data["density"])

# normalize/store stats for MAXMIN normalizing
dynViscnorm, con1dynVisc, con2dynVisc = scaler(dynVisc)
kinViscnorm, con1kinVisc, con2kinVisc = scaler(kinVisc)
surfTnorm, con1surfT, con2surfT = scaler(surfT)
densitynorm, con1density, con2density = scaler(density)

if "z" in NORMALIZE:
    stats = {
    "dynamic_viscosity": {"mean": con1dynVisc, "std": con2dynVisc},
    "kinematic_viscosity": {"mean": con1kinVisc,"std": con2kinVisc},
    "surface_tension": {"mean": con1surfT,"std": con2surfT},
    "density": {"mean": con1density,"std": con2density}
}
else:
    stats = {
    "dynamic_viscosity": {"max": con1dynVisc, "min": con2dynVisc},
    "kinematic_viscosity": {"max": con1kinVisc,"min": con2kinVisc},
    "surface_tension": {"max": con1surfT,"min": con2surfT},
    "density": {"max": con1density,"min": con2density}
}

# store normalized data
for idx in range(len(dynViscnorm)):
    data = {"dynamic_viscosity": float(dynViscnorm[idx]), "kinematic_viscosity": float(kinViscnorm[idx]), "surface_tension": float(surfTnorm[idx]),  "density": float(densitynorm[idx])}
    with open(f'{norm_path}/config_{(idx+1):04d}.json', 'w') as file:
        json.dump(data, file, indent=4)

# store statistics data
with open(f'{norm_path}/../statistics.json', 'w') as file:
    json.dump(stats, file, indent=4)