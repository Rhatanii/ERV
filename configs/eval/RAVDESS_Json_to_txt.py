import json
import os

json_file = "/mnt/ssd_hs/Dataset/RAVDESS/test.json"
txt_file = "/mnt/ssd_hs/Dataset/RAVDESS/test.txt"

with open(json_file, 'r') as f:
    data = json.load(f)

data_list = []

for item in data:
    # video_path = os.path.basename(item['vidpath'])
    video_path = item['vidpath'].replace('/mnt/ssd_sj/sejin/dataset/RAVDESS', '/mnt/ssd_hs/Dataset/RAVDESS')
    emotion = item['emotion']
    data_list.append(f"{video_path} {emotion}\n")

with open(txt_file, 'w') as f:
    f.writelines(data_list)