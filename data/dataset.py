from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import openpyxl
import torch
import json 

def read_json(file_path):
    with open(file_path, 'r') as rf:
        data = json.load(rf)
    return data

class VideoDataset(Dataset):
    def __init__(self, video_dataset_name, label_path, processor, modal,\
                dataset_src='/mnt/ssd_hs/Dataset'):
        self.dataset_src = dataset_src
        self.video_dataset_name = video_dataset_name



        if self.video_dataset_name =="DFEW":
            video_dir = os.path.join(self.dataset_src,video_dataset_name,'clip')
            labels = pd.read_csv(label_path, skiprows=1, header=None)
            self.labels = labels.values.tolist()
            self.video_paths = []
            self.video_ids = []

            for fname,label in self.labels:
                self.video_paths.append((os.path.join(video_dir, str(fname)+'.mp4'),label))
                self.video_ids.append(str(fname))
                    
        elif self.video_dataset_name =="MAFW":
            video_dir = os.path.join(self.dataset_src,video_dataset_name,'clips')
            self.video_paths = []
            self.video_ids = []

            with open(label_path,'r') as f:
                labels = f.readlines()
            
            self.labels = []
            for row in labels:
                fname, label = row.split()
                self.labels.append((fname, label))
            
            for fname, label in self.labels:
                self.video_paths.append((os.path.join(video_dir, fname),label))
                self.video_ids.append(str(fname))


        elif self.video_dataset_name =="MERR": 
            video_dir = os.path.join(self.dataset_src,'MER2023/test3')
            with open(label_path, 'r') as f:
                labels = json.load(f)
            
            self.labels = []
            for video_id in labels:
                self.labels.append((video_id, labels[video_id]['pseu_emotion']))
            
            self.video_paths = []
            for fname, label in self.labels:
                
                video_path = os.path.join(video_dir, str(fname)+'.avi')
                if os.path.exists(video_path):
                    self.video_paths.append((video_path, label))
                else:
                    video_path = video_path.replace('.avi', '.mp4')
                    if os.path.exists(video_path):
                        self.video_paths.append((video_path, label))
                    else:
                        print(f"Video file {fname} does not exist.")
                    
            
        elif self.video_dataset_name =="EMER":
            video_dir = os.path.join(self.dataset_src,'MER2023/train')
            self.video_paths = []
            self.video_ids = []

            for chunk in pd.read_csv(   label_path,
                                        skiprows=1,
                                        names=["names", "emotions", "subtitles", "reasons"],
                                        chunksize=1,  
                                    ):
                row = chunk.iloc[0]  
                self.video_paths.append((os.path.join(video_dir, str(row['names'])+'.avi'),row['emotions']))
                self.video_ids.append(str(row['names']))
        else:
            self.video_paths = [os.path.join(video_dir, fname) for fname in os.listdir(video_dir) if fname.endswith(('.mp4', '.mov'))]
            
        self.processor = processor
        self.modal = modal

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx].split('.')[0]
        video_path = self.video_paths[idx][0]
        label = self.video_paths[idx][1]
        video_tensor = self.processor['video'](video_path)
        audio_tensor = None
        if self.modal in ['video_audio', 'audio']:
            audio_tensor = self.processor['audio'](video_path)[0]
        else:
            audio_tensor = video_tensor.clone()
            

        return {
            'video_id': video_id,
            'video_path': video_path,
            'video_tensor': video_tensor,
            'audio_tensor': audio_tensor,
            'label': label
        }


