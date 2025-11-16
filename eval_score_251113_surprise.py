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
        emotion_count, correct, unknown, count_idx,unknown_list, confusion_matrix, pred_fname_sets, correct_pred_fname_sets = count_emotion_DFEW_251113_surprise(emotion_dict, total_lines, labels, check_wrong_samples)
        suprise_pred_ERV_set={'14940', '15909', '13474', '15256', '14926', '16305', '15155', '15998', '15982', '14472', '15533', '16166', '12985', '15825', '15921', '16077', '14375', '14120', '13585', '16354', '15416', '14519', '14676', '14717', '13892', '14514', '13512', '14424', '14374', '14079', '15984', '14849', '16356', '14481', '13394', '13854', '14719', '15242', '15098', '15320', '14421', '14682', '13539', '14907', '16310', '13942', '14788', '13561', '15664', '15879', '16116', '15286', '14909', '15773', '15884', '14663', '13442', '13605', '13407', '15906', '13768', '13541', '15662', '13572', '14700', '15093', '15914', '13180', '14322', '15874', '14487', '14798', '13890', '15924', '14621', '13712', '13889', '14688', '15185', '12712', '15187', '13908', '13710', '16186', '15276', '14031', '14279', '12595', '14575', '15794', '15066', '16260', '13833', '14076', '15567', '15962', '13667', '14383', '16033', '15290', '13895', '16316', '14251', '15467', '14477', '16002', '15452', '15591', '15684', '13754', '15740', '13972', '15204', '15592', '15870', '14877', '13356', '14211', '15055', '13506', '14734', '13536', '14624', '14334', '13546', '14630', '14812', '14027', '15964', '15480', '13631', '14337', '15724', '14531', '15910', '14381', '14509', '14067', '13933', '13794', '14438', '15958', '14353', '16322', '15774', '15417', '15490', '15099', '13885', '14615', '15627', '15659', '13459', '14001', '13363', '14756', '15131', '14902', '14833', '13228', '13571', '14801', '14373', '15514', '16009', '14249', '15755', '15214', '13643', '16321', '16144', '14893', '14747', '13277', '13648', '14518', '16177', '14770', '15322', '16309', '15328', '15751', '13408', '16229', '15967', '13905', '16162', '14162', '15887', '15387', '14637', '15413', '14920', '14587', '14000', '15252', '14152', '14707', '15022', '14930', '14245', '15833', '15272', '15040', '14344', '15381', '15430', '14619', '13416', '14116', '15313', '13992', '14226', '16208', '14708', '14028', '12928', '15125', '14623', '15599', '15095', '14672', '15442', '15550', '14325', '13997', '13753', '13580', '16192', '15201', '13038', '14035', '14085', '13882', '13502', '13498', '14687', '14491', '14450', '14650', '16091', '15087', '14605', '13832', '15428', '14517', '13743', '15265', '14808', '14501', '15418', '14345', '15409', '16366', '14022', '14608', '15085', '15540', '15653', '13849', '13810', '15548', '15151', '13431', '16105', '13374', '13831', '14166', '13820', '15233', '15630', '15027', '15786', '14557', '15761', '15782', '13657', '14836', '12874', '13575', '15046', '15051', '15096', '14863', '15411', '14410', '14548', '13490', '13981', '15805', '14136', '13718', '16262', '13978', '14340', '14792', '13295', '14379', '14400', '14175', '16092', '13809', '15965', '13973', '14760', '13681', '13918', '14403', '13654', '14961', '14771', '13937', '16085', '14462', '15146', '14530', '15500', '16004', '14534', '16042', '14611', '14873', '14882', '15525', '15429', '13607', '15283', '16236', '14826', '14938', '14885', '16100', '15090', '13713', '15928', '14772', '15745', '15445', '16037', '15815', '13782', '14444', '14824', '14573', '14652', '16034', '16067', '15048', '13646', '14592', '15330', '15031', '13549', '14736', '14432', '13503', '15335', '14583', '15017', '14567', '14045', '15327'}
        suprise_correct_ERV_set={'14517', '15533', '16004', '13667', '15755', '13833', '15146', '13643', '15490', '15099', '13490', '15525', '16208', '15022', '13646', '14885', '14045', '15265', '15031', '13942', '14700', '13580', '14630', '13394', '15330', '15051', '15313', '15066', '15540', '14477', '14000', '14251', '14893', '15187', '15664', '16037', '15773', '14501', '14337', '14245', '14652', '14079', '14717', '14708', '14353', '14481', '15098', '14573', '13561', '15027', '14487', '14534', '13890', '13605', '14226', '13831', '15322', '16236', '15256', '13743', '14877', '14637', '13654', '13882', '13972', '16356', '15984', '15761', '14688', '15242', '14770', '16002', '14462', '14882', '14344', '13546', '16354', '13885', '15514', '14687', '14824', '15445', '14909', '14621', '14676', '14085', '15055', '15480', '13498', '13502', '13648', '16162', '14211', '15096', '14421', '14788', '14760', '15442', '15409', '13981', '16321', '15740', '13363', '14031', '15413', '13572', '14403', '15592', '15958', '15328', '14530', '13753', '15417', '14067', '15928', '14028', '14410', '15684', '14279', '14920', '14438', '14432', '15201', '15924', '14509', '14444', '14619', '14961', '15906', '14491', '13820', '15233', '15627', '13536', '16177', '14624', '13997', '14930', '15982', '13459', '14136', '14801', '16366', '14771', '14605', '15550', '14812', '14849', '14592', '13407', '15095', '15429', '14798', '15659', '13854', '14383', '14514', '15185', '15965', '13442', '14531', '14022', '15825', '16092', '15276', '14719', '14650', '14166', '16042', '15327', '15204', '14373', '14175', '15093', '13908', '14926', '13539', '13933', '16316', '15967', '14583', '13810', '15452', '13607', '16009', '15387', '14381', '13571', '15874', '13973', '13416', '15548', '16322', '16077', '15320', '15879', '13710', '15430', '15833', '13918', '14450', '14076', '14152', '14736', '14747', '13794', '15786', '15151', '14027', '15335', '15087', '15774', '15567', '15921', '14334', '14792', '15283', '15131', '15085', '15662', '13978', '14035', '15887', '14379', '14756', '15046', '15418', '15252', '14116', '16116', '15794'}
        surprise_correct_MIGR_set={'14028', '15413', '15098', '15833', '14700', '14509', '14481', '13942', '15327', '14687', '13882', '13826', '14067', '14926', '15567', '13794', '15430', '14251', '15592', '15276', '14491', '14022', '14771', '14849', '14151', '16116', '13561', '15684', '15328', '15335', '16354', '13854', '14573', '16356', '15429', '14652', '14438', '15027', '14531', '15146', '15921', '16004', '13363', '14688', '16236', '14462', '14770', '13885', '14000', '15761', '14624', '15542', '15417', '15490', '13571', '14619', '15151', '16322', '14739', '15664', '15928', '15085', '16009', '13406', '14893', '14756', '14379', '15187', '15958', '14410', '14383', '13918', '14166', '13407', '15242', '14583', '15046', '14630', '14450', '15879', '13607', '13539', '14186', '15982', '15452', '15965', '13964', '15022', '15906', '15924', '14193', '14353', '15755', '14152', '13667', '13394', '15445', '13536', '15204', '13972', '13981', '14344', '15099', '15256', '15740', '13416', '15774', '14337', '13710', '14824', '14517', '15066', '13997', '14877', '15548', '14045', '14279', '15418', '14403', '14592', '13490', '13580', '13646', '14211', '14334', '15514', '13798', '15096', '13933', '13908', '14961', '15322', '13648', '13973', '14788', '14801', '14514', '15131', '15409', '14031', '14085', '16316', '15659', '14175', '15051', '16208', '13498', '15825', '15677', '14760', '15967', '15540', '16077', '14477', '14812', '15087', '13831', '14079', '13572', '15786', '14035', '16042', '15662', '14930', '14501', '15031', '14487', '15525', '14747', '14136', '14717', '13978', '14719', '13546', '13459', '14444', '14650', '15252', '14798', '16037', '16046', '15794', '15455', '16092', '14226', '14885', '13833', '15201', '15283', '16162', '15233', '13753', '16177', '13605', '15387', '15480'}
        # suprise_gap_set=suprise_correct_ERV_set - surprise_correct_MIGR_set
        # print('surprise gap set',suprise_gap_set)
        # fear_correct_ERV_set={'15386', '14832', '16328', '15551', '13224', '13017', '12927', '13516', '14215', '15954', '13040', '16108', '16174', '15976', '13969', '13335', '12718', '14549', '12669', '13559', '14294', '15303', '15820', '14718', '14613', '15536', '14977', '15603', '12711', '15239', '14185', '12583', '15415', '14012', '13931', '13256', '12585', '12902', '13179', '13062', '14649', '14576', '13553', '16300', '15354', '15398', '12620', '15184', '14721', '12963', '12844', '16125', '13535', '14281', '13106', '13705', '16200', '16342', '14087', '12634', '13707', '15674', '14600', '12645', '16364', '14670', '14524', '16261', '15583', '15065', '13449', '15996', '14694', '15594', '15464', '14536', '12949', '15183', '14119', '13494', '13399', '15573', '15275', '13870', '13129', '15616', '15912', '14069', '15134', '16153', '12652', '12943', '12847', '14520', '14014', '15907', '12818', '12997', '13280', '16237', '15053', '13558', '14839', '14929', '14545', '14398', '14923', '15934'}
        # fear_correct_MIGR_set={'16253', '14524', '15303', '15399', '16166', '12620', '13129', '14718', '14014', '16174', '14977', '13062', '12902', '15884', '14012', '14398', '14215', '15398', '15603', '13527', '15065', '15101', '12927', '14302', '13431', '13870', '13106', '14340', '14721', '15907', '12997', '14281', '13558', '13931', '15275', '12585', '16274', '14345', '13449', '15583', '16108', '16027', '13311', '13179', '13280', '15134', '15464', '14870', '14694', '13224', '12943', '14069', '14670', '15184', '13969', '15820', '15616', '15053', '16342', '14087', '15690', '14832', '15183', '14576', '15386', '14294', '16364', '16300', '13516', '12847', '16153', '16237', '14613', '16328', '14549', '15934', '14567', '15415', '15500', '13017', '14119', '13399', '16283', '12634', '14929', '12818', '15912', '14520', '15954', '14839', '12958', '12963', '12949', '12645', '13750', '16125', '14472', '13256', '12583', '13902', '13228', '15536', '15573', '13814', '13040', '13263', '13553', '15996', '12718', '12669', '16200', '12711', '15574', '14600', '15239', '12658', '13834', '13707', '13494', '14185', '14923', '15354', '13705', '14536'}
        # fear_gap_set= fear_correct_MIGR_set - fear_correct_ERV_set
        tmp_set={'15627', '14637', '14027', '13810', '14882', '14621', '14708', '15874', '13820', '15550', '14920', '14432', '15773', '13654', '14534', '15984', '13743', '14373', '14676', '14076', '16002', '15442', '13643', '14792', '14605', '15887', '14736', '15055', '15265', '15320', '15095', '14116', '15313', '14381', '15330', '15533', '13502', '15185', '14909', '13890', '14245', '14530', '14421', '16321', '13442', '16366', '15093'}
        emotion_count, correct, unknown, count_idx, unknown_list, confusion_matrix, pred_fname_sets = count_emotion_DFEW_251113_surprise_to_where(emotion_dict, total_lines, labels, tmp_set, check_wrong_samples)
        
    elif dataset_name == "MAFW":
        labels = extract_MAFW_labels(label_path, check_multi_emotion)
        if check_multi_emotion == True:
            correct, unknown, count_idx, unknown_list = count_emotion_MAFW(emotion_dict, total_lines, labels, check_multi_emotion, check_wrong_samples)
        else:
            emotion_count, correct, unknown, count_idx, unknown_list, confusion_matrix = count_emotion_MAFW(emotion_dict, total_lines, labels, check_multi_emotion, check_wrong_samples)
    elif dataset_name == "RAVDESS":
        labels = extract_RAVDESS_labels(label_path)
        emotion_count, correct, unknown, count_idx, unknown_list, confusion_matrix = count_emotion_RAVDESS(emotion_dict, total_lines, labels, check_wrong_samples)
    return emotion_count, correct, unknown, count_idx, unknown_list, confusion_matrix, pred_fname_sets, correct_pred_fname_sets

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
    dataset_name = "DFEW" # DFEW / MAFW / RAVDESS
    data_type="test" # train / test
    choose_id= "M-ERV-7B"  # Choose from result_folder_dict keys
    ckpt_id=1050 #"1044" or None
    temperature=None
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
    emotion_count, correct, unknown, count_idx, unknown_list, confusion_matrix, pred_fname_sets, correct_pred_fname_sets = extract_results(emotion_dict, total_lines, label_path, check_multi_emotion, check_wrong_samples)
    print(pred_fname_sets['fear'])
    # print(correct_pred_fname_sets['fear'])
    # 4. Print Results
    print_results(emotion_count, correct, unknown, count_idx, unknown_list, confusion_matrix, emotion_dict, check_multi_emotion, dataset_name)
    