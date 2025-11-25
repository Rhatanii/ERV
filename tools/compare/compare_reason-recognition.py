import json
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils_main import *



def rearrange_json(json_path):
    with open(json_path,'r') as f:
        gpt_lines = json.load(f)
    gpt_dict = {}
    for gpt_line in gpt_lines:
        video_id = list(gpt_line.keys())[0]
        gpt_result = gpt_line[video_id]['gpt_response']
        gpt_dict[video_id] = gpt_result
        
    return gpt_dict


if __name__=="__main__":
    ########### Configuration ###########
    dataset_name='DFEW'
    data_type ='test'
    modality='concat' 
    instruction_label_name =dataset_name 
    choose_ckpt=1044
    folder_name = "ERV-7B"
    
    if choose_ckpt is not None:
        folder_name += f"/checkpoint-{choose_ckpt}"

    if data_type == 'test':
        data_type_re = 'eval'
    else:
        data_type_re = data_type
    r1_omni_test_file = f"/mnt/ssd_hs/Exp/R1-Omni/results/{dataset_name}/{folder_name}/label_false/output_{data_type_re}5_all-video_audio.txt"
    
    if '/' in folder_name:
        folder_name_re='_'.join(folder_name.split('/'))
    else:
        folder_name_re = folder_name
        
    if instruction_label_name is None:
        gpt_test_json_file =r1_omni_test_file.replace('.txt','.json')
    else:
        gpt_test_json_file =r1_omni_test_file.replace('.txt',f'-instruct_{dataset_name}.json')



    if dataset_name == "DFEW":
        label_path=f"/mnt/ssd_hs/Dataset/DFEW/label/single_{data_type}set_5.csv"
        emotion_dict = {'happy': 1, 'sad': 2, 'neutral': 3, 'angry': 4, 'surprise': 5, 'disgust': 6, 'fear': 7}
    elif dataset_name == "MAFW":
        label_path=["/mnt/ssd_hs/Dataset/MAFW/Labels/single-set.xlsx"] 
        emotion_dict = {'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'neutral': 5, 'sadness': 6, 'surprise': 7, 'contempt': 8, 'anxiety': 9, 'helplessness':10, 'disappointment':11}
    reverse_emotion_dict = {v:k for k,v in emotion_dict.items()}
    
    # Reasoning & Recognition Output file
    aligned_r1_lines = rearrange_one_line(r1_omni_test_file)
    
    # gpt test json file
    gpt_dict = rearrange_json(gpt_test_json_file)

    if dataset_name == "DFEW":
        labels = pd.read_csv(label_path, skiprows=1, header=None)
        labels = labels.values.tolist()
        labels = extract_DFEW_labels(labels)
    elif dataset_name == "MAFW":
        labels = extract_MAFW_labels(label_path)

    
    # Compare
    quarter_table = {}
    quarter_id_table={}
    unknown  = 0
    for alinged_r1_line in aligned_r1_lines:
        # r1-omni test
        sample_id = alinged_r1_line.split('|')[0]
        try:
            r1_result = alinged_r1_line.split('|')[1].split('<answer>')[1].split('</answer>')[0].strip()
            
            if dataset_name == "DFEW":
                r1_result = match_emotion_key_DFEW(r1_result)
                gpt_result = match_emotion_key_DFEW(gpt_dict[sample_id])
            elif dataset_name == "MAFW":
                r1_result = match_emotion_key_MAFW(r1_result)
                gpt_result = match_emotion_key_MAFW(gpt_dict[sample_id])

            # Label
            emotion_gt_id =labels[sample_id][1] # emotion: (str)
            emotion_gt = reverse_emotion_dict[int(emotion_gt_id)]
            
            if r1_result == emotion_gt:
                r1_TF='True'
            else:
                r1_TF='False'
            
            if gpt_result == emotion_gt:
                gpt_TF='True'
            else:
                gpt_TF='False'
            
            if r1_result ==gpt_result:
                quarter_table_key='Explanation-Consistency'
                quarter_table[quarter_table_key] =quarter_table.get(quarter_table_key,0) + 1
                quarter_id_table[quarter_table_key] = quarter_id_table.get(quarter_table_key,[]) + [sample_id]
                
                
            quarter_table_key = f'R1-{r1_TF}/GPT-{gpt_TF}'
            quarter_table[quarter_table_key] =quarter_table.get(quarter_table_key,0) + 1
            quarter_id_table[quarter_table_key] = quarter_id_table.get(quarter_table_key,[]) + [sample_id]
        except:
            print(f"Error processing sample {sample_id}")
            print(alinged_r1_line)
            unknown += 1
            quarter_table_key = f'R1-False/GPT-False'
            quarter_table[quarter_table_key] =quarter_table.get(quarter_table_key,0) + 1
            quarter_id_table[quarter_table_key] = quarter_id_table.get(quarter_table_key,[]) + [sample_id]
    print(f"Unknown samples: {unknown}")


    total_num = sum(v for k, v in quarter_table.items() if k != "Explanation-Consistency")
    
    for k,v in quarter_table.items():
        print(f'{k}: {v/total_num:.2%}')
        

    output_folder =f'/mnt/ssd_hs/Exp/R1-Omni/results/{dataset_name}/{folder_name}/label_false/sample_id_list/{dataset_name}-{folder_name_re}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for k,v in quarter_table.items():
        name_k = k.replace('/','-')
        with open(f'{output_folder}/{name_k}.json','w', encoding='utf-8') as json_f:
            json.dump(quarter_id_table[k],json_f,indent=4, ensure_ascii=False)

