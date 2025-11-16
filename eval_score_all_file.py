'''
1. eval 폴더 읽기
2. 폴더 내 존재하는 모든 eval.txt 읽어서 성능 정리하기.

'''
import os
import pandas as pd

def match_emotion_key_DFEW(emotion_pred):
    if emotion_pred == 'sadness':
        emotion_pred = 'sad'
    elif 'happiness' in emotion_pred:
        emotion_pred = 'happy'
    elif 'frown' in emotion_pred:
        emotion_pred = 'sad'
    elif 'smile' in emotion_pred:
        emotion_pred = 'happy'
    elif 'sobbing' in emotion_pred:
        emotion_pred = 'sad'
    elif 'open mouth wide' in emotion_pred:
        emotion_pred = 'surprise'
    return emotion_pred

def match_emotion_key_MAFW(emotion_pred):
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
    elif 'frown' in emotion_pred:
        emotion_pred = 'sadness'
    elif 'smile' in emotion_pred:
        emotion_pred = 'happiness'
    elif 'sobbing' in emotion_pred:
        emotion_pred = 'sadness'
    elif 'open mouth wide' in emotion_pred:
        emotion_pred = 'surprise'
    elif 'contemptuous' in emotion_pred:
        emotion_pred = 'contempt'
    return emotion_pred

def rearrange_one_line(lines):
    '''
    Make result file one line for one id.
    '''
    total_lines =[]
    start_idx=0
    for idx, line in enumerate(lines):
        if line.split('|')[0].isdigit():
            if start_idx !=0:
                total_lines.append(tmp_line)
                start_idx=0
            start_idx +=1
            tmp_line = line
        else:
            tmp_line+=line
    return total_lines

def extract_DFEW_labels(labels: list):
    '''
    Convert list to dictionary.
    '''
    labels_dict = {}
    for idx, label in enumerate(labels):
        labels_dict[str(label[0])] = (str(label[0]), label[1])
    return labels_dict

def extract_MAFW_labels(label_path_list:list):
    '''
    MAFW label path -> (id,emotion_label) list
    /mnt/ssd_hs/Dataset/MAFW/Labels/single-set.xlsx (video_id, emotion_label)
    /mnt/ssd_hs/Dataset/MAFW/Labels/multi-set.xlsx (video_id, multi-emotion_label)
    '''
    total_dict = {}
    for label_path in label_path_list:
        df = pd.read_excel(label_path, skiprows=1)
        selected_columns = df.iloc[:, [0, 12]]
        
        
        for row in selected_columns.values:
            video_id = row[0].split('.mp4')[0]
            emotion_label = row[1]
            total_dict[video_id] = (video_id, emotion_label)

    return total_dict

    
def count_emotion_DFEW(emotion_dict, total_lines, labels: dict):
    emotion_count = [[0,0] for _ in range(len(emotion_dict.keys()))]
    confusion_matrix = [[0 for _ in range(len(emotion_dict.keys())+1)] for _ in range(len(emotion_dict.keys()))]
    correct=0
    unknown=0
    unknown_list = []
    
    for idx, line in enumerate(total_lines):
        try:
            fname=line.split('|')[0]

            if '<answer>' in line:
                emotion_pred = line.split('<answer>')[1].split('</answer>')[0].strip()
            else:
                emotion_pred = line.split('|')[1].strip()
                
            emotion_gt_id =labels[fname][1]
            
            # Emotion Key-Value matching
            emotion_pred = match_emotion_key_DFEW(emotion_pred)

            emotion_count[emotion_gt_id-1][1]+=1
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

def count_emotion_MAFW(emotion_dict, total_lines, labels: dict, check_multi_emotion: bool):
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
            else:
                emotion_pred = line.split('|')[1].strip()

            # Emotion Key-Value matching
            emotion_pred = match_emotion_key_MAFW(emotion_pred)
            
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
                    save_emotion = reverse_emotion_dict[emotion_gt_id]
                    save_path = f'./results/id_check/right/MAFW_wrong_{save_emotion}.txt'
                    
                    with open(save_path, 'a') as f:
                        f.write(f"{fname} {emotion_pred}\n")
                else:
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


if __name__ == "__main__":
    ## Configs
    dataset_name = "DFEW"
    check_multi_emotion= False # MAFW only for checking multi-emotion
    
    
    check_folder = f'/mnt/ssd_hs/Exp/R1-Omni/results/{dataset_name}'
    # 폴더 내 모든 txt 파일 가져오기
    all_txt_files =[]
    for (path,dir,files) in os.walk(check_folder):
        if len(files) > 0:
            print(path, files)
            for f in files:
                if f.endswith('.txt'):
                    all_txt_files.append('/'.join([path, f]))

    total_save_file= f'/mnt/ssd_hs/Exp/R1-Omni/results/{dataset_name}_all_251107.txt'
    with open(total_save_file, 'w') as outfile:
        for txt_file in all_txt_files:
            print(txt_file)
            check_file_name = txt_file.split(f'/mnt/ssd_hs/Exp/R1-Omni/results/{dataset_name}')[1]
            output_eval_path = txt_file

            try:

                if dataset_name == "DFEW":
                    label_path="/mnt/ssd_hs/Dataset/DFEW/label/single_testset_5.csv"
                    emotion_dict = {'happy': 1, 'sad': 2, 'neutral': 3, 'angry': 4, 'surprise': 5, 'disgust': 6, 'fear': 7}
                elif dataset_name == "MAFW":
                    label_path=["/mnt/ssd_hs/Dataset/MAFW/Labels/single-set.xlsx"] #,"/mnt/ssd_hs/Dataset/MAFW/Labels/multi-set.xlsx"]
                    emotion_dict = {'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'neutral': 5, 'sadness': 6, 'surprise': 7, 'contempt': 8, 'anxiety': 9, 'helplessness':10, 'disappointment':11}

                for tmp_emotion in emotion_dict.keys():
                    rm_path = f'./results/id_check/wrong/MAFW_wrong_{tmp_emotion}.txt'
                    rm_path2 = f'./results/id_check/right/MAFW_wrong_{tmp_emotion}.txt'
                    if os.path.exists(rm_path):
                        os.remove(rm_path)
                        print(f"Deleted file: {rm_path}")
                    if os.path.exists(rm_path2):
                        os.remove(rm_path2)
                        print(f"Deleted file: {rm_path2}")

                with open(output_eval_path,'r') as f:
                    lines = f.readlines()
                if len(lines) == 0:
                    continue
                # predict.txt 한줄로 변환.
                total_lines = rearrange_one_line(lines)


                # Load labels
                if dataset_name == "DFEW":
                    labels = pd.read_csv(label_path, skiprows=1, header=None)
                    labels = labels.values.tolist()
                    labels = extract_DFEW_labels(labels)
                    emotion_count, correct, unknown, count_idx, unknown_list, confusion_matrix = count_emotion_DFEW(emotion_dict, total_lines, labels)
                elif dataset_name == "MAFW":
                    labels = extract_MAFW_labels(label_path)
                    if check_multi_emotion == True:
                        correct, unknown, count_idx, unknown_list = count_emotion_MAFW(emotion_dict, total_lines, labels, check_multi_emotion)
                    else:
                        emotion_count, correct, unknown, count_idx, unknown_list, confusion_matrix = count_emotion_MAFW(emotion_dict, total_lines, labels, check_multi_emotion)


                print(unknown_list)
                print('Unknown::',unknown)
                print("WAR:",(correct/(count_idx+1))*100, correct, count_idx+1)
                
                
                if check_multi_emotion == False or dataset_name == "DFEW":
                    emotion_labels = list(emotion_dict.keys())

                    UAR=[]
                    for i, emotion_result in enumerate(emotion_count):
                        emotion=emotion_labels[i]
                        if emotion_result[1] == 0:
                            print(f"{emotion}:: {emotion_result[0]*100/(emotion_result[1]+1)} || {emotion_result[0]}/{emotion_result[1]} ")
                            uar_tmp = 0
                        else:
                            print(f"{emotion}:: {emotion_result[0]*100/emotion_result[1]} || {emotion_result[0]}/{emotion_result[1]} ")
                            uar_tmp = emotion_result[0]/emotion_result[1] *100
                        UAR.append(uar_tmp)

                    print("UAR:",sum(UAR)/len(UAR))
                    

                    # Draw Confusion Matrix
                    column_names = [f"{i}" for i in list(emotion_dict.keys())]
                    column_names.append("Unknown")
                    row_names = [f"{i}" for i in list(emotion_dict.keys())]
                    df = pd.DataFrame(confusion_matrix, columns=column_names, index=row_names)
                    print(df)
                    
                    ratio_confusion_matrix = confusion_matrix.copy()
                    for i in range(len(confusion_matrix)):
                        emotion_sum = sum(confusion_matrix[i])
                        for j in range(len(confusion_matrix[i])):
                            ratio_confusion_matrix[i][j] = (confusion_matrix[i][j] / emotion_sum)*100 if emotion_sum != 0 else 0
                    df = pd.DataFrame(ratio_confusion_matrix, columns=column_names, index=row_names)
                    print(df)
                    
                    print(f'{check_file_name} | WAR:',(correct/(count_idx+1))*100, correct, count_idx+1, f'UAR: {sum(UAR)/len(UAR)}', file=outfile)
            except:
                print(f"Error processing file: {output_eval_path}")
                continue