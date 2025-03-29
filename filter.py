import os
import os.path as osp
import json
import glob

matching_idx = []
para_paths = glob.glob(osp.join('original', 'parameters', "*.json"))

for filename in para_paths:
    with open(filename, 'r') as f:
        config = json.load(f)
        if config.get("density") == 3214 or config.get("surface_tension") == 19999 or config.get("density") == 13630:
            matching_idx.append(para_paths.index(filename))
        else:
            pass

print("Matching config IDs:", matching_idx)

"""
for idx in matching_idx:
    video_path = osp.join('original', 'videos', f'data_{idx:5d}.mp4')
    os.remove(video_path)
    print(f"Deleted: {video_path}")

new_video_paths = glob.glob(osp.join('original', 'videos', "*.mp4"))
new_para_paths = glob.glob(osp.join('original', 'parameters', "*.json"))

i = len(new_video_paths)
for video_path, para_path in zip(new_video_paths, new_para_paths):
    new_video_path = os.replace('video_path', f'data_{i:5d}.mp4')
    new_para_path = os.replace('original',  f'config_{i:5d}.json')
    os.rename(video_path, new_video_path)
    os.rename(para_path, new_para_path)
    i+=1
"""
