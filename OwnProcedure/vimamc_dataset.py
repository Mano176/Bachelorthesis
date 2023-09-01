from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import torch

class VimaMCDataset(Dataset):
    def __init__(self):
        self.length = len(os.listdir("./dataset"))


    def __len__(self):
        return self.length
    

    def __getitem__(self, idx):
        folder_path = "./dataset/"+str(idx+1).zfill(12)+"/"
        with open(folder_path+"prompt.txt", "r") as f:
            prompt = f.readline()
        with open(folder_path+"actions.txt", "r") as f:
            actions = [float(x) for x in f.readline().split(" ")]
        actions = {
            "cursor_x": actions[0],
            "cursor_y": actions[1],
            #"left_click_down": actions[2],
            #"left_click_up": actions[3],
            #"right_click_down": actions[4],
            #"right_click_up": actions[5]
        }
        obs = {
            "pov": torch.tensor(np.uint8(Image.open(folder_path+"obs.png").convert("RGB")))
        }
        asset = np.moveaxis(np.uint8(Image.open(folder_path+"asset.png").convert("RGB")), 2, 0)
        asset_segm = np.uint8(Image.open(folder_path+"asset_segm.png").convert("L"))
        prompt_assets = {
            "dragged_obj": {
                "rgb": {
                    "pov": asset,
                },
                "segm": {
                    "pov": asset_segm,
                    "obj_info": {
                        "obj_id": 0
                    }
                },
                "placeholder_type": "object"
            }
        }
        prompt_data = {"prompt": prompt, "prompt_assets": prompt_assets}
        return prompt_data, obs, actions