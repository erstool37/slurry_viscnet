import torch
import os.path as osp
import json
from statistics import mean, stdev
import wandb

# log10 > 0 to 1
def loginterscaler(lst):
    log_lst = torch.log10(torch.tensor(lst, dtype=torch.float32))
    min_val = log_lst.min()
    max_val = log_lst.max()
    scaled = (log_lst - min_val) / (max_val - min_val)
    return scaled, max_val.item(), min_val.item()

def loginterdescaler(scaled_lst, property):
    path = osp.dirname(osp.abspath(__file__))
    stat_path = osp.join(path, "../../dataset/CFDfluid/statistics.json")
    with open(stat_path, 'r') as file:
        data = json.load(file)
        max_val = torch.tensor(data[property]["max"], dtype=scaled_lst.dtype, device=scaled_lst.device)
        min_val = torch.tensor(data[property]["min"], dtype=scaled_lst.dtype, device=scaled_lst.device)
    log_val = scaled_lst * (max_val - min_val) + min_val
    return torch.pow(10, log_val)

# 0 to 1
def interscaler(lst): 
    max_list = max(lst)
    min_list = min(lst)
    scaled = [(x - min_list) / (max_list - min_list) for x in lst]
    return scaled, max_list, min_list

def interdescaler(scaled_lst, property):
    path = osp.dirname(osp.abspath(__file__))
    stat_path = osp.join(path,"../../dataset/CFDfluid/statistics.json")
    with open(stat_path, 'r') as file:
        data = json.load(file)
        max_lst = torch.tensor(data[property]["max"], dtype=scaled_lst.dtype, device=scaled_lst.device)
        min_lst = torch.tensor( data[property]["min"], dtype=scaled_lst.dtype, device=scaled_lst.device)
    return scaled_lst * (max_lst - min_lst) + min_lst
    
# zscore 0 to 1
def zscaler(lst):
    mean_val = mean(lst)
    std_val = stdev(lst)
    scaled = [(x - mean_val) / (2 * std_val) + 0.5 for x in lst] 
    return scaled, mean_val, std_val

def zdescaler(scaled_lst, property):
    path = osp.dirname(osp.abspath(__file__))
    stat_path = osp.join(path,"../../dataset/CFDfluid/statistics.json")
    with open(stat_path, 'r') as file:
        data = json.load(file)
        mean_lst = torch.tensor(data[property]["mean"], device=scaled_lst.device)
        std_lst = torch.tensor(data[property]["std"], device=scaled_lst.device)
    descaled = [(x - 0.5) * (2 * std_lst) + mean_lst for x in scaled_lst]
    return torch.tensor(descaled)

# log10 > zscore 0 to 1
def logzscaler(lst):
    log_lst = torch.log10(torch.tensor(lst, dtype=torch.float32)).tolist()
    mean_val = mean(log_lst)
    std_val = stdev(log_lst)
    scaled = [(x - mean_val) / (2 * std_val) + 0.5 for x in log_lst] 
    return scaled, mean_val, std_val

def logzdescaler(scaled_lst, property):
    path = osp.dirname(osp.abspath(__file__))
    stat_path = osp.join(path,"../../dataset/CFDfluid/statistics.json")
    with open(stat_path, 'r') as file:
        data = json.load(file)
        mean_lst = torch.tensor(data[property]["mean"], device=scaled_lst.device)
        std_lst = torch.tensor(data[property]["std"], device=scaled_lst.device)
    descaled = [(x - 0.5) * (2 * std_lst) + mean_lst for x in scaled_lst]
    return torch.pow(10, descaled)

# MEAN ABSOLUTE PERCENTAGE ERROR
def MAPEcalculator(pred, target, method):
    pred_den = loginterdescaler(pred[:,0], "density").unsqueeze(-1)
    pred_dynvisc = loginterdescaler(pred[:,1], "dynamic_viscosity").unsqueeze(-1)
    pred_surfT = loginterdescaler(pred[:,2], "surface_tension").unsqueeze(-1)

    target_den = loginterdescaler(target[:,0], "density").unsqueeze(-1)
    target_dynvisc = loginterdescaler(target[:,1], "dynamic_viscosity").unsqueeze(-1)
    target_surfT = loginterdescaler(target[:,2], "surface_tension").unsqueeze(-1)

    loss_mape_den = torch.mean((torch.abs(pred_den - target_den) / target_den)).unsqueeze(-1)
    loss_mape_dynvisc = torch.mean((torch.abs(pred_dynvisc - target_dynvisc) / target_dynvisc)).unsqueeze(-1)
    loss_mape_surfT = torch.mean((torch.abs(pred_surfT - target_surfT) / target_surfT)).unsqueeze(-1)

    wandb.log({f"MAPE {method} den %" : loss_mape_den * 100})
    wandb.log({f"MAPE {method} dynvisc %" : loss_mape_dynvisc * 100})
    wandb.log({f"MAPE {method} surfT %" : loss_mape_surfT * 100})