'''
여러 evaluation 조건에서 필요한 함수 정리.


'''
import random

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
    elif 'contemptuous' in emotion_pred:
        emotion_pred = 'contempt'
    # elif 'frown' in emotion_pred:
    #     emotion_pred = 'sadness'
    # elif 'smile' in emotion_pred:
    #     emotion_pred = 'happiness'
    # elif 'sobbing' in emotion_pred:
    #     emotion_pred = 'sadness'
    # elif 'open mouth wide' in emotion_pred:
    #     emotion_pred = 'surprise'

    return emotion_pred


def make_fine_question(pred_emotion):
    emotion_connection_dict = {"anger":["anger", "disgust", "contempt", "anxiety"],
                               "happiness":["happiness","neutral", "contempt"],
                               "neutral":["disgust", "neutral", "anxiety", "helplessness", "disappointment"],
                               "sadness":["fear", "sadness", "anxiety","helplessness", "disappointment"],
                               "surprise":["fear", "surprise", "anxiety"]}
    tmp_emo_list = emotion_connection_dict[pred_emotion]
    random.shuffle(tmp_emo_list)
    round2_candidates=','.join(tmp_emo_list)
    
    return round2_candidates
