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

# sanity check
MAXMIN = 

1000
if con1dynVisc == con2dynVisc:
    dynViscnorm = 1
    con1dynVisc += 1e-4
if con1kinVisc == con2kinVisc:
    kinViscnorm += 1e-5
    con1kinVisc += 1e-5
if con1surfT == con2surfT:
    surfTnorm += 1e-5
    con1surfT += 1e-3
if con1density == con2density:
    densitynorm += 1e-5
    con1density += 100
if con2dynVisc == 0:
    dynViscnorm += 1e-5
    con1dynVisc -= 1e-4
    con2dynVisc += 1e-4
if con2kinVisc == 0:   
    kinViscnorm += 1e-5
    con1kinVisc -= 1e-5 
    con2kinVisc += 1e-5
if con2surfT == 0:
    surfTnorm += 1e-5
    con1surfT -= 1e-3
    con2surfT += 1e-3
if con2density == 0:
    densitynorm += 1e-5
    con1density -= 100
    con2density += 100

if "z" in NORMALIZE:
    stats = {
        "dynamic_viscosity": {"mean": float(con1dynVisc), "std": float(con2dynVisc)},
        "kinematic_viscosity": {"mean": float(con1kinVisc), "std": float(con2kinVisc)},
        "surface_tension": {"mean": float(con1surfT), "std": float(con2surfT)},
        "density": {"mean": float(con1density), "std": float(con2density)}
    }
else:
    stats = {
        "dynamic_viscosity": {"max": float(con1dynVisc), "min": float(con2dynVisc)},
        "kinematic_viscosity": {"max": float(con1kinVisc), "min": float(con2kinVisc)},
        "surface_tension": {"max": float(con1surfT), "min": float(con2surfT)},
        "density": {"max": float(con1density), "min": float(con2density)}
    }

# store normalized data
for idx in range(len(dynViscnorm)):
    data = {"dynamic_viscosity": float(dynViscnorm[idx]), "kinematic_viscosity": float(kinViscnorm[idx]), "surface_tension": float(surfTnorm[idx]),  "density": float(densitynorm[idx])}
    with open(f'{norm_path}/config_{(idx+1):04d}.json', 'w') as file:
        json.dump(data, file, indent=4)

# store statistics data
with open(f'{norm_path}/../statistics.json', 'w') as file:
    json.dump(stats, file, indent=4)