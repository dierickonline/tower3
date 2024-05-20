import pandas as pd
import numpy as np
import torch
import ast
from PIL import Image
from torch.utils.data import Dataset

# Pytorch Dataset Class, inherited from torch.utils.data.Dataset
class TowerDataset(Dataset):
       
    def __init__(self, root, transforms=None):
        self.root = root
        self.labels_df = pd.read_csv(self.root + 'all_patches_annotated.csv')  

        self.labels_df['objects'] = self.labels_df['objects'].apply(ast.literal_eval)

        self.labels_df['boxes'] = self.labels_df['objects'].apply(lambda x: [item['box'] for item in x])
        self.labels_df['classes'] = self.labels_df['objects'].apply(lambda x: [int(item['class']) for item in x])


        # self.labels_df['boxes'] = self.labels_df['boxes'].apply(ast.literal_eval)
        # self.labels_df['classes'] = self.labels_df['classes'].apply(ast.literal_eval)
        self.transforms  = transforms

    def __len__(self):
        return self.labels_df.shape[0]

    def __getitem__(self, idx):

        
        #self.labels_df['boxes'] = self.labels_df['boxes'].apply(ast.literal_eval)
        #self.labels_df['classes'] = self.labels_df['classes'].apply(ast.literal_eval)
        
        img_path =  self.root + self.labels_df['image'][idx]
        img = Image.open(img_path)
        
        num_objs = len(self.labels_df.iloc[idx, 4])

        

        boxes = torch.as_tensor(self.labels_df.at[idx, 'boxes'], dtype=torch.float32) if len(self.labels_df.at[idx, 'boxes']) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        labels = []
        labels = torch.as_tensor(self.labels_df.at[idx, 'classes'], dtype=torch.int64)

        image_id = torch.tensor([idx])
        
        iscrowd = torch.zeros(num_objs, dtype=torch.int64)


        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) 
        
             
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
    
