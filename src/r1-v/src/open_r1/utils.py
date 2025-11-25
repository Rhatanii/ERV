import json



def match_emotion_key_DFEW(emotion_pred):
    emotion_pred = emotion_pred.lower()
    emotion_pred = emotion_pred.replace('\n', '')
    if emotion_pred == 'sadness':
        emotion_pred = 'sad'
    elif 'happiness' in emotion_pred:
        emotion_pred = 'happy'
    elif 'anger' in emotion_pred:
        emotion_pred = 'angry'

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

    return emotion_pred
