import torch
import os.path as osp
import json
from statistics import mean, stdev
import wandb
import importlib

# log10 > 0 to 1
def loginterscaler(lst):
    log_lst = torch.log10(torch.tensor(lst, dtype=torch.float32))
    min_val = log_lst.min()
    max_val = log_lst.max()
    scaled = (log_lst - min_val) / (max_val - min_val)
    return scaled, max_val.item(), min_val.item()

def loginterdescaler(scaled_lst, property, path):
    root = osp.dirname(osp.abspath(__file__))
    stat_path = osp.join(root, "../..", path, "statistics.json")
    with open(stat_path, 'r') as file:
        data = json.load(file)
        max_val = torch.tensor(data[property]["max"], dtype=scaled_lst.dtype, device=scaled_lst.device)
        min_val = torch.tensor(data[property]["min"], dtype=scaled_lst.dtype, device=scaled_lst.device)
    log_val = scaled_lst * (max_val - min_val) + min_val
    return torch.pow(10, log_val)

# 0 to 1
def interscaler(lst): 
    lst = torch.tensor(lst, dtype=torch.float32)
    min_val = lst.min()
    max_val = lst.max()
    scaled = (lst - min_val) / (max_val - min_val)
    return scaled, max_val, min_val

def interdescaler(scaled_lst, property, path):
    root = osp.dirname(osp.abspath(__file__))
    stat_path = osp.join(root, "../..", path, "statistics.json")
    with open(stat_path, 'r') as file:
        data = json.load(file)
        max_val = torch.tensor(data[property]["max"], dtype=scaled_lst.dtype, device=scaled_lst.device)
        min_val = torch.tensor( data[property]["min"], dtype=scaled_lst.dtype, device=scaled_lst.device)
    return scaled_lst * (max_val - min_val) + min_val
    
# zscore 0 to 1
def zscaler(lst):
    lst = torch.tensor(lst, dtype=torch.float32)
    mean_val = lst.mean()
    std_val = lst.std()
    scaled = [(x - mean_val) / (2 * std_val) + 0.5 for x in lst] 
    return scaled, mean_val, std_val

def zdescaler(scaled_lst, property, path):
    root = osp.dirname(osp.abspath(__file__))
    stat_path = osp.join(root, "../..", path, "statistics.json")
    with open(stat_path, 'r') as file:
        data = json.load(file)
        mean_val = torch.tensor(data[property]["mean"], dtype=scaled_lst.dtype, device=scaled_lst.device)
        std_val = torch.tensor(data[property]["std"], dtype=scaled_lst.dtype, device=scaled_lst.device)
    descaled = (scaled_lst - 0.5) * (2 * std_val) + mean_val
    return descaled

# log10 > zscore 0 to 1
def logzscaler(lst):
    log_lst = torch.log10(torch.tensor(lst, dtype=torch.float32))
    mean_val = log_lst.mean()
    std_val = log_lst.std()
    scaled = (log_lst - mean_val) / (2 * std_val) + 0.5
    return scaled, mean_val, std_val

def logzdescaler(scaled_lst, property, path):
    root = osp.dirname(osp.abspath(__file__))
    stat_path = osp.join(root, "../..", path, "statistics.json")
    with open(stat_path, 'r') as file:
        data = json.load(file)
        mean_val = torch.tensor(data[property]["mean"], dtype=scaled_lst.dtype, device=scaled_lst.device)
        std_val = torch.tensor(data[property]["std"], dtype=scaled_lst.dtype, device=scaled_lst.device)
    descaled = (scaled_lst - 0.5) * (2 * std_val) + mean_val
    return torch.pow(10, descaled)

def noscaler(lst):
    lst = torch.tensor(lst, dtype=torch.float32)
    return lst, torch.tensor(1.0), torch.tensor(1.0)

def nodescaler(lst, property, path):
    lst = torch.tensor(lst, dtype=torch.float32)
    return lst

# MEAN ABSOLUTE PERCENTAGE ERROR
def MAPEcalculator(pred, target, descaler, method, path):
    utils = importlib.import_module("utils")
    descaler = getattr(utils, descaler)
    
    pred_den = descaler(pred[:,0], "density", path).unsqueeze(-1)
    pred_dynvisc = descaler(pred[:,1], "dynamic_viscosity", path).unsqueeze(-1)
    pred_surfT = descaler(pred[:,2], "surface_tension", path).unsqueeze(-1)

    target_den = descaler(target[:,0], "density", path).unsqueeze(-1)
    target_dynvisc = descaler(target[:,1], "dynamic_viscosity", path).unsqueeze(-1)
    target_surfT = descaler(target[:,2], "surface_tension", path).unsqueeze(-1)

    loss_mape_den = torch.mean((torch.abs(pred_den - target_den) / target_den)).unsqueeze(-1)
    loss_mape_dynvisc = torch.mean((torch.abs(pred_dynvisc - target_dynvisc) / target_dynvisc)).unsqueeze(-1)
    loss_mape_surfT = torch.mean((torch.abs(pred_surfT - target_surfT) / target_surfT)).unsqueeze(-1)

    wandb.log({f"MAPE {method} den %" : loss_mape_den * 100})
    wandb.log({f"MAPE {method} dynvisc %" : loss_mape_dynvisc * 100})
    wandb.log({f"MAPE {method} surfT %" : loss_mape_surfT * 100})