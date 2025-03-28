from torch.utils.data import IterableDataset, Dataset
import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F

class VideoDataset(Dataset):
    def __init__(self, video_paths, para_paths, frame_num, time):
        '''Initialize dataset'''
        self.video_paths = video_paths
        self.para_paths = para_paths
        self.frame_limit = frame_num * time
        self.frame_count = 30

    def __getitem__(self, index):
        frames = self.__loadvideo__(self.video_paths[index], self.frame_limit)
        parameters = self.__loadparameters__(self.para_paths[index])
        return frames, parameters

    def __loadvideo__(self, video_path, frame_limit):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened() and len(frames) < self.frame_count:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR → RGB
                frames.append(frame)
        cap.release()

        frames = np.array(frames, dtype=np.float32) / 255.0 # required only for no masked

        return torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) # (T, C, H, W)
    
    def __loadparameters__(self, para_path):
        with open(para_path, 'r') as file:
            data = json.load(file)
            density = data["density"]
            dynVisc = float(data["dynamic_viscosity"])
            surfT = float(data["surface_tension"])
            kinVisc = float(data["kinematic_viscosity"])
            
        return torch.tensor([density, dynVisc, surfT, kinVisc], dtype=torch.float32)

    def __len__(self):
        return len(self.video_paths)
    
    """
    def __iter__(self):
        for video_path, para_path in zip(self.video_paths, self.para_paths):
            frames = self.__loadvideo__(video_path, self.frame_limit)
            parameters = self.__loadparameters__(para_path)
            yield frames, parameters
    """