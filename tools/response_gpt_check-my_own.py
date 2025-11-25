
import json
import argparse
from openai import OpenAI
import sys
import os
import pandas as pd
import random
from tqdm import tqdm
from utils import gpt_response

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils_main import *



def order_R1Omni_result(reason_result_path, dataset_name):
    '''
    R1-Omni result file -> 'id': {'reason': *, 'emotion': *} dict
    '''
    lines = rearrange_one_line(reason_result_path)
        
    R1Omni_dict= {}
    for line in lines:
        line = line.split('|')
        sample_id = line[0]
        try:
            reason = line[1].split('<think>')[1].split('</think>')[0].strip()
        except IndexError:
            # print(sample_id, line)
            reason = ''
        vis_reason = line[1].split('<vis_desc>')[1].split('</vis_desc>')[0].strip() if '<vis_desc>' in line[1] else ''
        aud_reason = line[1].split('<aud_desc>')[1].split('</aud_desc>')[0].strip() if '<aud_desc>' in line[1] else ''
        concat_reason = vis_reason + ' ' + aud_reason + ' ' + reason
        emotion = line[1].split('<answer>')[-1].replace('</answer>', '').strip()
        
        try:
            if dataset_name == "DFEW":
                emotion = match_emotion_key_DFEW(emotion)
            elif dataset_name == "MAFW":
                emotion = match_emotion_key_MAFW(emotion)
            else:
                raise ValueError('Unknown dataset name')
        except ValueError:
            emotion = 'neutral'  # Default to neutral if unrecognized

        R1Omni_dict[sample_id] = {'reason': reason, 'vis_reason': vis_reason, 'aud_reason': aud_reason, 'concat_reason': concat_reason, 'pred_emotion': emotion}

    return R1Omni_dict

def extract_DFEW_label_dict(label_path: str, emotion_dict: dict):
    '''
    DFEW label path -> 'id':{gt_emotion: *} dict
    '''
    labels = pd.read_csv(label_path, skiprows=1, header=None)
    labels = labels.values.tolist()
    reverse_emotion_dict = {v: k for k, v in emotion_dict.items()}
    labels_dict = {}
    for idx, label in enumerate(labels):
        emotion = reverse_emotion_dict[label[1]]
        labels_dict[str(label[0])] = {'gt_emotion': emotion}
    return labels_dict

def extract_MAFW_label_dict(label_path: str, emotion_dict: dict):
    '''
    MAFW label path -> 'id':{gt_emotion: *} dict
    '''
    with open(label_path,'r') as f:
        labels = f.readlines()
    
    label_dict = {}
    for row in labels:
        video_id, emotion_label = row.split()
        video_id = video_id.split('.mp4')[0]
        emotion_label = match_emotion_key_MAFW(emotion_label)
        
        label_dict[video_id] = {'gt_emotion': emotion_label}

    return label_dict



def extract_wrong_sample(R1Omni_dict, label_dict, remove_emotion_list: list, random_max: int):
    '''
    Extract wrong sample id.
    remove_emotion_list: emotion to be not considered
    '''
    def shuffle_dict(original_dict):
        items = list(original_dict.items())
        random.shuffle(items)
        return dict(items)
    
    new_remove_emotion_list = []
    for emotion in remove_emotion_list:
        emotion = match_emotion_key_MAFW(emotion)
        new_remove_emotion_list.append(emotion)
        
    if random_max is not None:
        R1Omni_dict = shuffle_dict(R1Omni_dict)
    
    wrong_sample_dict = {}
    count_dict = {}
    
    for sample_id, content in R1Omni_dict.items():
        gt_emotion = label_dict[sample_id]['gt_emotion']
        pred_emotion = content['pred_emotion']
    
        if pred_emotion != gt_emotion:
            if gt_emotion not in new_remove_emotion_list:
                if random_max is not None:
                    if gt_emotion not in count_dict.keys():
                        count_dict[gt_emotion] = 1
                    else:
                        if count_dict[gt_emotion] >= random_max:
                            continue
                        else:
                            count_dict[gt_emotion] +=1
                wrong_sample_dict[sample_id] = {
                    'reason': content['reason'],
                    'pred_emotion': pred_emotion,
                    'gt_emotion': gt_emotion                
                }
    return wrong_sample_dict
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="R1-Omni additional GPT reasoning")
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset_name')
    parser.add_argument('--instruction_label_name', type=str, required=True, help='instruction dataset')
    parser.add_argument('--data_type', type=str, required=True, help='dataset type train ,test ')
    parser.add_argument('--nshard', type=int, default=1, help='number of shards')
    parser.add_argument('--shard_id', type=int, default=0, help='shard id')
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    args = parser.parse_args()
    
    ############# Configs ################
    dataset_name=args.dataset_name
    data_type = args.data_type
    model_id =args.model_id
    instruction_label_name=args.instruction_label_name

    
    # 1. Load results path
    if data_type =="test":
        reason_result_path = f'/mnt/ssd_hs/Exp/R1-Omni/results/{dataset_name}/{model_id}/label_false/output_eval5_all-video_audio.txt'
    else:
        reason_result_path = f'/mnt/ssd_hs/Exp/R1-Omni/results/{dataset_name}/{model_id}/label_false/output_train5_all-video_audio.txt'
    
    # 2. Load label path and emotion dict
    if dataset_name == "DFEW":
        label_path=f"/mnt/ssd_hs/Dataset/DFEW/label/single_{data_type}set_5.csv"
        emotion_dict = {'happy': 1, 'sad': 2, 'neutral': 3, 'angry': 4, 'surprise': 5, 'disgust': 6, 'fear': 7}
    elif dataset_name == "MAFW":
        label_path=f"/mnt/ssd_hs/Dataset/MAFW/Train & Test Set/single/no_caption/set_5/{data_type}.txt"
        emotion_dict = {'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'neutral': 5, 'sadness': 6, 'surprise': 7, 'contempt': 8, 'anxiety': 9, 'helplessness':10, 'disappointment':11}
    
    output_path = reason_result_path.replace('.txt',f'-instruct_{instruction_label_name}-{args.shard_id}.json')

    ######################################

    # (Sharding) reason_dict 
    R1Omni_dict = order_R1Omni_result(reason_result_path, dataset_name)
    R1_lengths = len(R1Omni_dict)
    shard_size = (R1_lengths + args.nshard - 1) // args.nshard  
    R1_items = list(R1Omni_dict.items())
    sharded_dicts = [dict(R1_items[i:i + shard_size]) for i in range(0, R1_lengths, shard_size)]
    shard_R1_Omni_dict = sharded_dicts[args.shard_id]

    # Load labels
    if dataset_name == "DFEW":
        label_dict = extract_DFEW_label_dict(label_path, emotion_dict)
    elif dataset_name == "MAFW":
        label_dict = extract_MAFW_label_dict(label_path, emotion_dict)

    try:
        with open(output_path,'r') as rf:
            json_data = json.load(rf)
            start_idx = len(json_data)
            print(f"Already exist {start_idx} samples")
    except:
        start_idx = 0
        
    
    # instruction settings
    total_list=[]
    try:
        for idx, (sample_k,sample_v) in tqdm(enumerate(shard_R1_Omni_dict.items()),total=len(shard_R1_Omni_dict),desc="Generating GPT responses"):
            if idx < start_idx:
                continue
            json_data={}
            
            sample_id = sample_k
            reason = sample_v['reason']


            if dataset_name == "DFEW":
                emotion_str_list =['happy', 'sad', 'neutral', 'angry', 'surprise', 'disgust', 'fear']
                emotion_str = ', '.join(emotion_str_list)
                INSTRUCTION_CHECK = f"Read the reasoning content and respond with the appropriate emotion in {emotion_str}. Reply with only the emotion word."
            elif dataset_name == "MAFW":
                emotion_str_list =['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'contempt', 'anxiety', 'helplessness', 'disappointment']
                emotion_str = ', '.join(emotion_str_list)
                INSTRUCTION_CHECK = f"Read the reasoning content and respond with the appropriate emotion in {emotion_str}. Reply with only the emotion word."
            
            instruction = INSTRUCTION_CHECK + f"\nReason: {reason}\nAnswer Emotion:"
            output = gpt_response(instruction, gpt_version='gpt-4.1-mini')
            print(output)
            
            # filter out of bound emotions.
            if dataset_name =="MAFW":
                if output not in ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'contempt', 'anxiety', 'helplessness', 'disappointment']:
                    output = 'neutral'
            elif dataset_name == "DFEW":
                if output not in ['happy', 'sad', 'neutral', 'angry', 'surprise', 'disgust', 'fear']:
                    output = 'neutral'
                    
            json_data[sample_id] = {
                'reason': reason,
                'gpt_response': output
            }
            total_list.append(json_data)

        with open(output_path,'a') as wf:
            json.dump(total_list, wf, ensure_ascii=False, indent=4)  
    except:
        print(f"Error at {idx} sample id: {sample_id}")
        with open(output_path,'a') as wf:
            json.dump(total_list, wf, ensure_ascii=False, indent=4)

