
import os
import shutil
import re
import pandas as pd
from tqdm import tqdm

def rearrange_one_line(output_eval_path):
    """
    Read a multi-line result file and merge all lines belonging to the same sample ID
    into a single line. Each ID starts with a digit followed by '|'.

    Args:
        output_eval_path (str): Path to the result file.

    Returns:
        list[str]: A list where each element corresponds to one sample ID (one line).
    """
    with open(output_eval_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = []
    current_line = ""

    for line in lines:
        # If line starts with a numeric ID (e.g., "10337|...")
        if line.strip() and line.split('|')[0].isdigit():
            # Save the previous accumulated line
            if current_line:
                total_lines.append(current_line.strip())
            # Start a new line
            current_line = line.strip()
        else:
            # Append continuation lines
            current_line += " " + line.strip()

    if current_line:
        total_lines.append(current_line.strip())

    return total_lines


def match_emotion_key_DFEW(emotion_pred):
    emotion_pred = emotion_pred.lower()
    emotion_pred = emotion_pred.replace('\n', '')
    if emotion_pred == 'sadness':
        emotion_pred = 'sad'
    elif 'happiness' in emotion_pred:
        emotion_pred = 'happy'
    elif 'anger' in emotion_pred:
        emotion_pred = 'angry'

    if emotion_pred not in ['happy', 'sad', 'angry', 'neutral', 'fear', 'disgust', 'surprise']:
        raise ValueError(f"Invalid emotion label: {emotion_pred}")
    return emotion_pred

def match_emotion_key_MAFW(emotion_pred):
    emotion_pred = emotion_pred.lower()
    emotion_pred = emotion_pred.replace('\n', '')
    if emotion_pred =='angry':
        emotion_pred = 'anger'
    elif emotion_pred == 'sad':
        emotion_pred = 'sadness'
    elif emotion_pred == 'anxious':
        emotion_pred = 'anxiety'
    elif emotion_pred == 'helpless':
        emotion_pred = 'helplessness'
    elif emotion_pred == 'disappointed':
        emotion_pred = 'disappointment'
    elif 'happy' in emotion_pred:
        emotion_pred = 'happiness'
    elif 'contemptuous' in emotion_pred:
        emotion_pred = 'contempt'

    if emotion_pred not in ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'contempt', 'anxiety', 'helplessness', 'disappointment']:
        raise ValueError(f"Invalid emotion label: {emotion_pred}")
    return emotion_pred


def extract_DFEW_labels(labels: list):
    '''
    Convert list to dictionary.
    '''
    labels_dict = {}
    for idx, label in enumerate(labels):
        labels_dict[str(label[0])] = (str(label[0]), label[1])
    return labels_dict

def extract_MAFW_labels(label_path_list:list, check_multi_emotion: bool=False):
    '''
    MAFW label path -> (id,emotion_label) list
    /mnt/ssd_hs/Dataset/MAFW/Labels/single-set.xlsx (video_id, emotion_label)
    /mnt/ssd_hs/Dataset/MAFW/Labels/multi-set.xlsx (video_id, multi-emotion_label)
    '''
    total_dict = {}

    emotion_dict = {'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'neutral': 5, 'sadness': 6, 'surprise': 7, 'contempt': 8, 'anxiety': 9, 'helplessness':10, 'disappointment':11}
    label_path_list.sort(key=lambda x: 0 if "single/" in x else 1)
    
    for label_path in label_path_list:
        df = pd.read_excel(label_path, skiprows=1)
        selected_columns = df.iloc[:, [0, 12]] # video_id, emotion_label
        
        for row in selected_columns.values:
            video_id = row[0].split('.mp4')[0]
            emotion_label = row[1]
            total_dict[video_id] = (video_id, emotion_label)

    return total_dict




def count_emotion_DFEW(emotion_dict, total_lines, labels: dict, check_wrong_samples: bool=False):
    emotion_count = [[0,0] for _ in range(len(emotion_dict.keys()))]
    confusion_matrix = [[0 for _ in range(len(emotion_dict.keys())+1)] for _ in range(len(emotion_dict.keys()))]
    correct=0
    unknown=0
    unknown_list = []
    
    for idx, line in enumerate(total_lines):
        try:
            fname=line.split('|')[0]

            if '<answer>' in line:
                if '</answer>' not in line:
                    # Handle cases where the closing tag is missing
                    emotion_pred = re.search(r'<answer>(.*?)$', line).group(1).strip()
                    print(emotion_pred)
                else:
                    emotion_pred = line.split('<answer>')[1].split('</answer>')[0].strip()
            elif '<|im_end|>' in line:
                emotion_pred = line.split('|')[1].split('<|im_end|>')[0].strip()
            else:
                emotion_pred = line.split('|')[1].strip()
            emotion_gt_id =labels[fname][1]
            
            # Emotion Key-Value matching
            emotion_count[emotion_gt_id-1][1]+=1
            emotion_pred = match_emotion_key_DFEW(emotion_pred)

            emotion_pred_id = emotion_dict[emotion_pred]
            confusion_matrix[emotion_gt_id-1][emotion_pred_id-1]+=1

            if emotion_pred_id == emotion_gt_id:
                correct+=1
                emotion_count[emotion_pred_id-1][0]+=1
        except:
            unknown+=1
            unknown_list.append(emotion_pred)
            confusion_matrix[emotion_gt_id-1][-1]+=1
            
    return emotion_count, correct, unknown, idx, unknown_list, confusion_matrix



def count_emotion_MAFW(emotion_dict, total_lines, labels: dict, check_multi_emotion: bool, check_wrong_samples: bool=False):
    """
        Check accuracy.
        Args:
            emotion_dict: dict {emotion_label: emotion_id}
            total_lines: list of lines from eval.txt ['1|description1', '2|description2', ...]
            labels: dict {video_id: (video_id, emotion_label)}
    
    """
    # [[correct, total], ...]
    emotion_count = [[0,0] for _ in range(len(emotion_dict.keys()))]
    confusion_matrix = [[0 for _ in range(len(emotion_dict.keys())+1)] for _ in range(len(emotion_dict.keys()))]
    correct=0
    unknown=0
    unknown_list = []
    reverse_emotion_dict = {v: k for k, v in emotion_dict.items()}

    for idx, line in enumerate(total_lines):
        try:
            fname=line.split('|')[0]

            if '<answer>' in line:
                emotion_pred = line.split('<answer>')[1].split('</answer>')[0].strip()
            elif '<|im_end|>' in line:
                emotion_pred = line.split('|')[1].split('<')[0].strip()
            else:
                emotion_pred = line.split('|')[1].strip()

            # Predicted Emotion
            emotion_pred = match_emotion_key_MAFW(emotion_pred)
            # Label Emotion
            emotion_gt_id =labels[fname][1] 
            

            # Check Multi-label emotion
            if type(emotion_gt_id) == str:
                tmp_multi_emotion_gt_list = emotion_gt_id.split(',')
                tmp_multi_emotion_gt_list = [int(i) for i in tmp_multi_emotion_gt_list]
            else:
                tmp_multi_emotion_gt_list = [int(emotion_gt_id)]

            
            if len(tmp_multi_emotion_gt_list) > 1:
                emotion_pred_id = emotion_dict[emotion_pred]
                if emotion_pred_id in tmp_multi_emotion_gt_list:
                    correct+=1
            else:
                emotion_count[emotion_gt_id-1][1]+=1
                emotion_pred_id = emotion_dict[emotion_pred]
                confusion_matrix[emotion_gt_id-1][emotion_pred_id-1]+=1

                if emotion_pred_id == emotion_gt_id:
                    correct+=1
                    emotion_count[emotion_pred_id-1][0]+=1
                else:
                    if check_wrong_samples == True:
                        save_emotion = reverse_emotion_dict[emotion_gt_id]
                        save_path = f'./results/id_check/wrong/MAFW_wrong_{save_emotion}.txt'
                        
                        with open(save_path, 'a') as f:
                            f.write(f"{fname} {emotion_pred}\n")
        except:
            unknown+=1
            unknown_list.append(emotion_pred)
            if len(tmp_multi_emotion_gt_list) == 1:
                confusion_matrix[emotion_gt_id-1][-1]+=1
    
    if check_multi_emotion == True:
        return correct, unknown, idx, unknown_list
    else:   
        return emotion_count, correct, unknown, idx, unknown_list, confusion_matrix





# Split eval results into True and False Confusion Matrices
def split_eval_results_DFEW(emotion_dict, total_lines, labels: dict):
    """
        Returns: 
        True_confusion_matrix:
        False_confusion_matrix:
    """
    emotion_count = [[0,0] for _ in range(len(emotion_dict.keys()))]
    confusion_matrix = [[0 for _ in range(len(emotion_dict.keys())+1)] for _ in range(len(emotion_dict.keys()))]
    true_confusion_matrix = [[[] for _ in range(len(emotion_dict.keys())+1)] for _ in range(len(emotion_dict.keys())+1)]
    false_confusion_matrix = [[[] for _ in range(len(emotion_dict.keys())+1)] for _ in range(len(emotion_dict.keys())+1)]
    
    
    correct=0
    unknown=0
    unknown_list = []
    
    for idx, line in enumerate(total_lines):
        try:
            fname=line.split('|')[0]

            if '<answer>' in line:
                emotion_pred = line.split('<answer>')[1].split('</answer>')[0].strip()
            elif '<|im_end|>' in line:
                emotion_pred = line.split('|')[1].split('<|im_end|>')[0].strip()
            else:
                emotion_pred = line.split('|')[1].strip()
            emotion_gt_id =labels[fname][1]
            
            # Emotion Key-Value matching
            emotion_pred = match_emotion_key_DFEW(emotion_pred)

            emotion_count[emotion_gt_id-1][1]+=1
            emotion_pred_id = emotion_dict[emotion_pred]


            if emotion_pred_id == emotion_gt_id:
                true_confusion_matrix[emotion_gt_id-1][emotion_pred_id-1].append(fname)
                correct+=1
                emotion_count[emotion_pred_id-1][0]+=1
            else:
                false_confusion_matrix[emotion_gt_id-1][emotion_pred_id-1].append(fname)
        except:
            unknown+=1
            unknown_list.append(emotion_pred)

            
    return emotion_count, correct, unknown, idx, unknown_list, true_confusion_matrix, false_confusion_matrix


def split_eval_results_MAFW(emotion_dict, total_lines, labels: dict):
    """
        Returns: 
        True_confusion_matrix:
        False_confusion_matrix:
    """
    emotion_count = [[0,0] for _ in range(len(emotion_dict.keys()))]
    true_confusion_matrix = [[[] for _ in range(len(emotion_dict.keys())+1)] for _ in range(len(emotion_dict.keys())+1)]
    false_confusion_matrix = [[[] for _ in range(len(emotion_dict.keys())+1)] for _ in range(len(emotion_dict.keys())+1)]
    
    correct=0
    unknown=0
    unknown_list = []
    reverse_emotion_dict = {v: k for k, v in emotion_dict.items()}

    for idx, line in tqdm(enumerate(total_lines)):
        try:
            fname=line.split('|')[0]

            if '<answer>' in line:
                emotion_pred = line.split('<answer>')[1].split('</answer>')[0].strip()
            elif '<|im_end|>' in line:
                emotion_pred = line.split('|')[1].split('<')[0].strip()
            else:
                emotion_pred = line.split('|')[1].strip()

            # Emotion Key-Value matching
            emotion_pred = match_emotion_key_MAFW(emotion_pred)

            emotion_gt_id =labels[fname][1]
            # if reverse_emotion_dict[int(emotion_gt_id)] == emotion_pred:
            #     continue
            
            ########################################################################
            # want_to_check_gt_emotion = "anger"
            # want_to_check_pred_emotion = "surprise"
            # if emotion_pred == want_to_check_pred_emotion and emotion_gt_id == emotion_dict[want_to_check_gt_emotion]:
            #     print(fname)
            #     src_path = f"/mnt/ssd_hs/Dataset/MAFW/clips/{fname}.mp4"
            #     dest_path = f"/mnt/ssd_hs/Exp/R1-Omni/results_tmp/MAFW_test/gt_{want_to_check_gt_emotion}/pred_{want_to_check_pred_emotion}/{fname}.mp4"
            #     os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            #     shutil.copy(src_path,dest_path)
            ########################################################################
            # Check Multi-label emotion

            if type(emotion_gt_id) == str:
                tmp_multi_emotion_gt_list = emotion_gt_id.split(',')
                tmp_multi_emotion_gt_list = [int(i) for i in tmp_multi_emotion_gt_list]
            else:
                tmp_multi_emotion_gt_list = [int(emotion_gt_id)]

            
            if len(tmp_multi_emotion_gt_list) > 1:
                emotion_pred_id = emotion_dict[emotion_pred]
                if emotion_pred_id in tmp_multi_emotion_gt_list:
                    correct+=1
    
            else:
                emotion_count[emotion_gt_id-1][1]+=1
                emotion_pred_id = emotion_dict[emotion_pred]

                if emotion_pred_id == emotion_gt_id:
                    correct+=1
                    emotion_count[emotion_pred_id-1][0]+=1
                    true_confusion_matrix[emotion_gt_id-1][emotion_pred_id-1].append(fname)
                else:
                    false_confusion_matrix[emotion_gt_id-1][emotion_pred_id-1].append(fname)
        except:
            unknown+=1
            unknown_list.append(emotion_pred)


    return emotion_count, correct, unknown, idx, unknown_list, true_confusion_matrix, false_confusion_matrix
