import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, video_paths, para_paths, frame_limit):
        '''Initialize dataset'''
        self.video_paths = video_paths
        self.para_paths = para_paths
        self.frame_limit = frame_limit

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, index):
        video_path = self.video_paths[index]
        para_path = self.para_paths[index]
        
        frames = self.__loadvideo__(video_path, self.frame_limit)
        frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)  # (T, C, H, W)

        parameters = self.__loadparameters__(para_path)

        return frames, parameters
    
    def __loadvideo__(self, video_path, frame_limit=32):
        cap = cv2.VideoCapture(video_path)
        frames = []
    
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or len(frames) >= frame_limit:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB
            frames.append(frame)
        cap.release()

        while len(frames) < frame_limit:
            frames.append(np.zeros_like(frames[0]))  # Add empty frames if needed
    
        return np.array(frames, dtype=np.uint8) # (T, C, H, W)
    
    def __loadparameters__(self, para_path):
        parameters = torch.tensor(np.load(para_path), dtype=torch.float32).squeeze(0)  # (1, N)

        return parameters