import json
import os.path as osp
import glob
import math
from statistics import mean, stdev
import numpy as np
import torch

def logscaler(lst): # 0~1 scaled data
    log_list = lst  # Apply log10 element-wise
    max_list = max(log_list)
    min_list = min(log_list)
    scaled = [(x - min_list) / (max_list - min_list) for x in log_list]
    return scaled, max_list, min_list

def zscaler(lst):
    mean_val = mean(lst)
    std_val = stdev(lst)
    scaled = [(x - mean_val) / (2 * std_val) + 0.5 for x in lst] 
    return scaled, mean_val, std_val

def logdescaler(scaled_lst, property):
    path = osp.dirname(osp.abspath(__file__))
    stat_path = osp.join(path,"../../dataset/CFDfluid/statistics.json")
    with open(stat_path, 'r') as file:
        data = json.load(file)
        max_lst = torch.tensor(data[property]["max"], dtype=scaled_lst.dtype, device=scaled_lst.device)
        min_lst = torch.tensor( data[property]["min"], dtype=scaled_lst.dtype, device=scaled_lst.device)
    return scaled_lst * (max_lst - min_lst) + min_lst

"""
def zdescaler(scaled_lst, property):
    path = osp.dirname(osp.abspath(__file__))
    stat_path = osp.join(path,"../../dataset/CFDfluid/statistics.json")
    with open(stat_path, 'r') as file:
        data = json.load(file)
        mean_lst = torch.tensor(data[property]["mean"], device=scaled_lst.device)
        std_lst = torch.tensor(data[property]["std"], device=scaled_lst.device)
    descaled = [(x - 0.5) * (2 * std_lst) + mean_lst for x in scaled_lst].

    return torch.tensor(descaled)
"""

def zdescaler(scaled_tensor, property):
    path = osp.dirname(osp.abspath(__file__))
    stat_path = osp.join(path, "../../dataset/CFDfluid/statistics.json")
    with open(stat_path, 'r') as file:
        data = json.load(file)
    mean = torch.tensor(data[property]["mean"], dtype=scaled_tensor.dtype, device=scaled_tensor.device)
    std = torch.tensor(data[property]["std"], dtype=scaled_tensor.dtype, device=scaled_tensor.device)
    return (scaled_tensor - 0.5) * (2 * std) + mean

# Start normalizing
DATA_ROOT = "dataset/CFDfluid" # use "../../dataset/CFDfluid" for creating stats.
PARA_SUBDIR = "parameters"
NORM_SUBDIR = "parametersNorm"
para_paths = sorted(glob.glob(osp.join(DATA_ROOT, PARA_SUBDIR, "*.json")))
norm_path = osp.join(DATA_ROOT, NORM_SUBDIR)

dynVisc = []
kinVisc = [] 
surfT = []
density = []

# call variables
for path in para_paths:
    with open(path, 'r') as file:
        data = json.load(file)
        dynVisc.append(data["dynamic_viscosity"])
        kinVisc.append(data["kinematic_viscosity"])
        surfT.append(data["surface_tension"])
        density.append(data["density"])

# normalize
dynViscnorm, maxdynVisc, mindynVisc = logscaler(dynVisc)
kinViscnorm, maxkinVisc, minkinVisc = logscaler(kinVisc)
surfTnorm, maxsurfT, minsurfT = logscaler(surfT)
densitynorm, maxdensity, mindensity = logscaler(density)

# store normalized data
for idx in range(len(dynViscnorm)):
    data = {"dynamic_viscosity": dynViscnorm[idx], "kinematic_viscosity": kinViscnorm[idx], "surface_tension": surfTnorm[idx],  "density": densitynorm[idx]}
    with open(f'{norm_path}/config_{(idx+1):04d}.json', 'w') as file:
        json.dump(data, file, indent=4)

# store statistics data
"""
stats = {
    "dynamic_viscosity": {"mean": maxdynVisc, "std": mindynVisc},
    "kinematic_viscosity": {"mean": maxkinVisc,"std": minkinVisc},
    "surface_tension": {"mean": maxsurfT,"std": minsurfT},
    "density": {"mean": maxdensity,"std": mindensity}
}
"""
stats = {
    "dynamic_viscosity": {"max": maxdynVisc, "min": mindynVisc},
    "kinematic_viscosity": {"max": maxkinVisc,"min": minkinVisc},
    "surface_tension": {"max": maxsurfT,"min": minsurfT},
    "density": {"max": maxdensity,"min": mindensity}
}

with open(f'{norm_path}/../statistics.json', 'w') as file:
    json.dump(stats, file, indent=4)