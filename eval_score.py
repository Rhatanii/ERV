'''
1. Load label path and emotion dict
2. Load eval.txt, Rearrange to one line per sample
3. Count Results
4. Print Results

'''
import pandas as pd
pd.set_option('display.max_columns', None)  # 모든 열 출력
pd.set_option('display.width', 1000)       # 출력 폭 설정

from utils_main import *

def load_label_path_and_emotion_dict(dataset_name, data_type, check_multi_emotion):
    if dataset_name == "DFEW":
        label_path=f"/mnt/ssd_hs/Dataset/DFEW/label/single_{data_type}set_5.csv"
        emotion_dict = {'happy': 1, 'sad': 2, 'neutral': 3, 'angry': 4, 'surprise': 5, 'disgust': 6, 'fear': 7}
    elif dataset_name == "MAFW":
        if check_multi_emotion == True:
            label_path=["/mnt/ssd_hs/Dataset/MAFW/Labels/single-set.xlsx", "/mnt/ssd_hs/Dataset/MAFW/Labels/multi-set.xlsx"]
        else:
            label_path=["/mnt/ssd_hs/Dataset/MAFW/Labels/single-set.xlsx"] 
        emotion_dict = {'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'neutral': 5, 'sadness': 6, 'surprise': 7, 'contempt': 8, 'anxiety': 9, 'helplessness':10, 'disappointment':11}
    elif dataset_name == "RAVDESS":
        label_path = "/mnt/ssd_hs/Dataset/RAVDESS/test.txt"
        emotion_dict = {'happy': 1, 'surprised': 2, 'neutral': 3, 'angry': 4, 'disgust': 5, 'sad': 6, 'fearful': 7, 'calm': 8}

    return label_path, emotion_dict

def extract_results(emotion_dict, total_lines, label_path, check_multi_emotion, check_wrong_samples):
    emotion_count = None
    confusion_matrix= None
    if dataset_name == "DFEW":
        labels = pd.read_csv(label_path, skiprows=1, header=None)
        labels = labels.values.tolist()
        labels = extract_DFEW_labels(labels)
        emotion_count, correct, unknown, count_idx, unknown_list, confusion_matrix = count_emotion_DFEW(emotion_dict, total_lines, labels, check_wrong_samples)
    elif dataset_name == "MAFW":
        labels = extract_MAFW_labels(label_path, check_multi_emotion)
        if check_multi_emotion == True:
            correct, unknown, count_idx, unknown_list = count_emotion_MAFW(emotion_dict, total_lines, labels, check_multi_emotion, check_wrong_samples)
        else:
            emotion_count, correct, unknown, count_idx, unknown_list, confusion_matrix = count_emotion_MAFW(emotion_dict, total_lines, labels, check_multi_emotion, check_wrong_samples)
    elif dataset_name == "RAVDESS":
        labels = extract_RAVDESS_labels(label_path)
        emotion_count, correct, unknown, count_idx, unknown_list, confusion_matrix = count_emotion_RAVDESS(emotion_dict, total_lines, labels, check_wrong_samples)
    return emotion_count, correct, unknown, count_idx, unknown_list, confusion_matrix

def print_results(emotion_count, correct, unknown, count_idx, unknown_list, confusion_matrix, emotion_dict, check_multi_emotion, dataset_name):
    print('Unknown:',unknown_list)
    print('Total Unknown Count:', len(unknown_list))
    print("===================================")
    
    if check_multi_emotion == False or dataset_name == "DFEW":
        emotion_labels = list(emotion_dict.keys())

        UAR=[]
        for i, emotion_result in enumerate(emotion_count):
            emotion=emotion_labels[i]
            if emotion_result[1] == 0:
                print(f"{emotion}:: {emotion_result[0]*100/(emotion_result[1]+1):.3f} || {emotion_result[0]}/{emotion_result[1]} ")
                uar_tmp = 0
            else:
                print(f"{emotion}:: {emotion_result[0]*100/emotion_result[1]:.3f} || {emotion_result[0]}/{emotion_result[1]} ")
                uar_tmp = emotion_result[0]/emotion_result[1] *100
            UAR.append(uar_tmp)
        
        print("===================================")
        print(f"WAR: {(correct/(count_idx+1))*100:.3f}", correct, count_idx+1)
        print(f"UAR: {sum(UAR)/len(UAR):.3f}")
        print("===================================")
        

        # Draw Confusion Matrix
        column_names = [f"{i}" for i in list(emotion_dict.keys())]
        column_names.append("Unknown")
        row_names = [f"{i}" for i in list(emotion_dict.keys())]
        df = pd.DataFrame(confusion_matrix, columns=column_names, index=row_names).round(3)
        print(df)
        
        ratio_confusion_matrix = confusion_matrix.copy()
        for i in range(len(confusion_matrix)):
            emotion_sum = sum(confusion_matrix[i])
            for j in range(len(confusion_matrix[i])):
                ratio_confusion_matrix[i][j] = (confusion_matrix[i][j] / emotion_sum)*100 if emotion_sum != 0 else 0
        df = pd.DataFrame(ratio_confusion_matrix, columns=column_names, index=row_names).round(3)
        print(df)
    else:
        print("===============MAFW Compound ================")
        print(f"WAR: {(correct/(count_idx+1))*100:.3f}", correct, count_idx+1)
        print("===================================")



if __name__ == "__main__":
    ###################### Configs ######################
    dataset_name = "MAFW" # DFEW / MAFW / RAVDESS
    data_type="test" # train / test
    choose_id= "Ablation_TS_only_MIGR"  # Choose from result_folder_dict keys
    ckpt_id=1000 #"1044" or None 
    temperature=0.3
    result_folder_dict = {"R1-7B": "R1-7B",
                          "ERV-7B": "ERV-7B",
                          "R1-0.5B": "R1-0.5B",
                          "ERV-0.5B": "ERV-0.5B",
                          "EMER-SFT-7B": "EMER-SFT-7B",
                          "EMER-SFT-0.5B": "EMER-SFT-0.5B",
                          "M-ERV-7B": "M-ERV-7B",
                          "TRI-AV-MI-SFT-7B": "TRI-AV-MI-SFT-7B",
                          "TRI-AV-NO-MI-SFT-7B": "TRI-AV-NO-MI-SFT-7B",
                          "MERR-SFT-7B": "MERR-SFT-7B",
                          "Baseline-0.5B": "Baseline-0.5B",
                          "Ablation-TRI-AV-SFT-7B_wo_TS": "Ablation-TRI-AV-SFT-7B_wo_TS",
                          "results-11-08-0324-AV-EMER-SFT-0.5B-af_a_tsa-epoch2-G16-lr1e-6-bs2-ga2/checkpoint-900": "results-11-08-0324-AV-EMER-SFT-0.5B-af_a_tsa-epoch2-G16-lr1e-6-bs2-ga2/checkpoint-900"
                          
                          }
    ###################### Options ######################
    check_wrong_samples = False # Prediction 틀리는 Sample 저장.
    check_multi_emotion= False # MAFW only for checking multi-emotion
    
    if ckpt_id is None:
        result_forder_name=choose_id
    else:
        result_forder_name=f"{choose_id}/checkpoint-{ckpt_id}"
        
    if temperature is not None:
        output_eval_path =f"/mnt/ssd_hs/Exp/R1-Omni/results/{dataset_name}/{result_forder_name}/label_false/output_eval5_all-video_audio-tmp{temperature}.txt"
    else:
        output_eval_path =f"/mnt/ssd_hs/Exp/R1-Omni/results/{dataset_name}/{result_forder_name}/label_false/output_eval5_all-video_audio.txt"

    ###################### Evaluation ######################
    # 1. Load label path and emotion dict
    label_path, emotion_dict = load_label_path_and_emotion_dict(dataset_name, data_type, check_multi_emotion)
    
    # 2. Load eval.txt, Rearrange to one line per sample
    if dataset_name == "RAVDESS":
        total_lines = rearrange_one_line_ravdess(output_eval_path)
    else:
        total_lines = rearrange_one_line(output_eval_path)

    # 3. Count Results
    emotion_count, correct, unknown, count_idx, unknown_list, confusion_matrix = extract_results(emotion_dict, total_lines, label_path, check_multi_emotion, check_wrong_samples)

    # 4. Print Results
    print_results(emotion_count, correct, unknown, count_idx, unknown_list, confusion_matrix, emotion_dict, check_multi_emotion, dataset_name)
    