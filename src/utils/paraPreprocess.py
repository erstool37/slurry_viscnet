import json
import os.path as osp
import glob
import math
from statistics import mean, stdev

DATA_ROOT = "../../dataset/CFDfluid/"
PARA_SUBDIR = "parameters"
NORM_SUBDIR = "parametersNorm"

para_paths = sorted(glob.glob(osp.join(DATA_ROOT, PARA_SUBDIR, "*.json")))
norm_path = osp.join(DATA_ROOT, NORM_SUBDIR)

def logscaler(lst): # 0~1 scaled data
    log_list = [math.log10(x) for x in lst]  # Apply log10 element-wise
    max_list = max(log_list)
    min_list = min(log_list)
    scaled = [(x - min_list) / (max_list - min_list) for x in log_list]
    return scaled

def zscaler(lst):
    mean_val = mean(lst)
    std_val = stdev(lst)
    scaled = [(x - mean_val) / (2 * std_val) + 0.5 for x in lst] 
    return scaled

dynVisc = []
kinVisc = [] 
surfT = []
density = []

# collocate each variable
for path in para_paths:
    with open(path, 'r') as file:
        data = json.load(file)

        dynVisc.append(data["dynamic_viscosity"])
        kinVisc.append(data["kinematic_viscosity"])
        surfT.append(data["surface_tension"])
        density.append(data["density"])

# normalize
dynViscnorm = logscaler(dynVisc)
kinViscnorm = logscaler(kinVisc)
surfTnorm = zscaler(surfT)
densitynorm = zscaler(density)

# stock
for idx in range(len(dynViscnorm)):
    data = {"density": density[idx], "dynamic_viscosity": dynViscnorm[idx], "surface_tension": surfTnorm[idx], "kinematic_viscosity": kinViscnorm[idx]}
    with open(f'{norm_path}/config_{(idx+1):04d}.json', 'w') as file:
        json.dump(data, file, indent=4)