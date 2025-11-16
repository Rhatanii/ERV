import json


def read_json_to_dict(file_path):
    """
    Read json file and convert it to a dictionary.
    Args:
        file_path (str): Path to the json file.
    Returns:
        dict: Dictionary representation of the json file.
    """
    with open(file_path, 'r') as rf:
        data_dict = json.load(rf)
    return data_dict

def match_emotion_key_DFEW(emotion_pred):
    if emotion_pred == 'sadness':
        emotion_pred = 'sad'
    elif 'happiness' in emotion_pred:
        emotion_pred = 'happy'
    # elif 'frown' in emotion_pred:
    #     emotion_pred = 'sad'
    # elif 'smile' in emotion_pred:
    #     emotion_pred = 'happy'
    # elif 'sobbing' in emotion_pred:
    #     emotion_pred = 'sad'
    # elif 'open mouth wide' in emotion_pred:
    #     emotion_pred = 'surprise'
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

def match_emotion_EMOBERT(emotion_pred):
    goemotions_to_target = {
    "admiration": "happiness",         # 긍정적 감정 → happiness
    "amusement": "happiness",          # 유쾌한 감정 → happiness
    "anger": "anger",                  # 직접 대응
    "annoyance": "anger",              # 짜증 → anger
    "approval": "happiness",           # 긍정 피드백 → happiness
    "caring": "happiness",             # 배려 → happiness
    "confusion": "anxiety",           # 혼란 → anxiety
    "curiosity": "neutral",            # 탐구 → 중립적
    "desire": "happiness",             # 욕망 → 긍정적 기대
    "disappointment": "disappointment",# 직접 대응
    "disapproval": "contempt",         # 반감 → contempt (멸시 포함 가능)
    "disgust": "disgust",              # 직접 대응
    "embarrassment": "anxiety",        # 불안 계열 → anxiety
    "excitement": "happiness",         # 긍정 고양 감정
    "fear": "fear",                    # 직접 대응
    "gratitude": "happiness",          # 감사 → happiness
    "grief": "helplessness",           # 깊은 슬픔 → helplessness로 매핑
    "joy": "happiness",                # 직접 대응
    "love": "happiness",               # 일반적 긍정 → happiness
    "nervousness": "anxiety",          # 직접 대응
    "optimism": "happiness",           # 긍정 기대
    "pride": "happiness",              # 자기 긍정 감정
    "realization": "neutral",          # 깨달음 → 중립
    "relief": "happiness",             # 해방감 → 행복
    "remorse": "sadness",              # 후회 → sadness
    "sadness": "sadness",              # 직접 대응
    "surprise": "surprise",            # 직접 대응
    "neutral": "neutral"               # 직접 대응
    }
    emotion = goemotions_to_target.get(emotion_pred, emotion_pred)
    return emotion
