# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import re
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from datasets import load_dataset, load_from_disk, Features,Sequence,Value
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, BitsAndBytesConfig

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, HumanOmniVLGRPOTrainer, HumanOmniVLGRPO_TEXT_Trainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

local_rank = int(os.environ.get("LOCAL_RANK", 0))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    EMOTION_DEVICE = torch.device(f"cuda:{local_rank}")
else:
    EMOTION_DEVICE = torch.device("cpu")



_EMOTION_MODEL_ID ="/mnt/ssd_hs/Exp/R1-Omni/src/text_emotion_classifier/emotion_trainer-EMER-MERR-gpt_ver_label_11-balance-epoch10-bs64-ga1-lr2e-05/checkpoint-180"
EMOTION_TOKENIZER = AutoTokenizer.from_pretrained(_EMOTION_MODEL_ID)
EMOTION_MODEL  = AutoModelForSequenceClassification.from_pretrained(_EMOTION_MODEL_ID)
EMOTION_MODEL.eval()

EMOTION_MODEL.to(EMOTION_DEVICE)

EMOTION_MODEL_DICT={
    "happiness": 0,
    "anger": 1,
    "disgust": 2,
    "fear": 3,
    "sadness": 4,
    "surprise": 5,
    "neutral": 6,
    "anxiety": 7,
    "helplessness": 8,
    "disappointment": 9,
    "contempt": 10,
}
REVERSE_EMOTION_MODEL_DICT = {v: k for k, v in EMOTION_MODEL_DICT.items()}

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "think", "chain", "consistency","think_with_hint", "think_emotion_label", "think_top1", "think_half"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    reference_model_switch: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use the reference model for training"},
    )
    data_ratio: Optional[float] = field(
        default=1.0,
        metadata={"help": "Ratio of data to use for training"},
    )
    test_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Test dataset name."},
    )
    lora_enabled: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to enable LoRA for training"},
    )
    bnb_enabled: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to enable BitsAndBytes for training"},
    )

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


####################
gpt_keyword_path = "/mnt/ssd_hs/Dataset/R1-Omni/clues/emotion_clues.json"
emotion_keyword_dict = read_json_to_dict(gpt_keyword_path)
confusion_matrix_path = "/mnt/ssd_hs/Dataset/R1-Omni/clues/emotion_confusion_matrix_max.json"
confusion_matrix_dict = read_json_to_dict(confusion_matrix_path)
TF_IDF_PATH= "/mnt/ssd_hs/Dataset/R1-Omni/clues/tfidf_scores.json"
tfidf_dict = read_json_to_dict(TF_IDF_PATH)
####################

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

def think_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    sub_rewards = []
    videos = kwargs.get("video", "")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    except_list =[]
    # For batch processing
    total_student_think= []
    for i, cont in enumerate(contents):
        try:
            total_student_think.append(re.search(r'<think>(.*?)</think>', cont).group(1).strip())
        except:
            total_student_think.append("")
            except_list.append(i)
    em_inputs = EMOTION_TOKENIZER(total_student_think,
                                  return_tensors="pt",
                                    padding=True,
                                    truncation=True).to(EMOTION_DEVICE)
    with torch.no_grad():
        logits = EMOTION_MODEL(**em_inputs).logits
    probs = F.softmax(logits, dim=1).squeeze() # argmax
    topk = torch.topk(probs, k=1) 
    top_indices = topk.indices.tolist()
    
    text_emotion_list=[]
    for index_list in top_indices:
        tmp_list = []
        for index in index_list:
            tmp_list.append(REVERSE_EMOTION_MODEL_DICT[index])
        text_emotion_list.append(tmp_list) # numgeneration 개수, 각 내부는 topk개씩.
        
    # text_emotion_list =[]
    # for top_label_list in top_labels:
    #     tmp_top_label_list = []
    #     for emotion in top_label_list:
    #         tmp_top_label_list.append(emotion)
    #     text_emotion_list.append(tmp_top_label_list)
        
    
    ground_truth_emotion_list = []

    # 예측 gt
    for sol in solution:
        # Extract answer from solution if it has think/answer tags
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        ground_truth_emotion_list.append(match_emotion_key_MAFW(ground_truth))


    for i, (top_emotion_list, gt_emo) in enumerate(zip(text_emotion_list, ground_truth_emotion_list)):
        if i in except_list:
            rewards.append(0.0)
        else:
            if gt_emo in top_emotion_list:
                reward = 1.0 #top_value[0]
                rewards.append(reward)
            # elif gt_emo in top_emotion_list[1]:
            #     reward = top_value[1]
            #     rewards.append(reward)
            else:
                rewards.append(0.0)

    return rewards

def think_half_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    sub_rewards = []
    videos = kwargs.get("video", "")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    except_list =[]
    # For batch processing
    total_student_think= []
    for i, cont in enumerate(contents):
        try:
            total_student_think.append(re.search(r'<think>(.*?)</think>', cont).group(1).strip())
        except:
            total_student_think.append("")
            except_list.append(i)
    em_inputs = EMOTION_TOKENIZER(total_student_think,
                                  return_tensors="pt",
                                    padding=True,
                                    truncation=True).to(EMOTION_DEVICE)
    with torch.no_grad():
        logits = EMOTION_MODEL(**em_inputs).logits
    probs = F.softmax(logits, dim=1).squeeze()
    topk = torch.topk(probs, k=2)
    top_indices = topk.indices.tolist()
    top_values = topk.values.tolist()
    
    top_labels=[]
    for index_list in top_indices:
        tmp_list = []
        for index in index_list:
            tmp_list.append(EMOTION_MODEL.config.id2label[index])
        top_labels.append(tmp_list) # numgeneration 개수, 각 내부는 topk개씩.
        
    text_emotion_list =[]
    for top_label_list in top_labels:
        tmp_top_label_list = []
        for top_label in top_label_list:
            emotion = match_emotion_EMOBERT(top_label)
            tmp_top_label_list.append(emotion)
        text_emotion_list.append(tmp_top_label_list)
        
    
    ground_truth_emotion_list = []

    # 예측 gt
    for sol in solution:
        # Extract answer from solution if it has think/answer tags
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        ground_truth_emotion_list.append(match_emotion_key_MAFW(ground_truth))


    for i, (top_emotion_list, top_value, gt_emo) in enumerate(zip(text_emotion_list, top_values, ground_truth_emotion_list)):
        if i in except_list:
            rewards.append(0.0)
        else:
            if gt_emo in top_emotion_list:
                reward = 0.5 #top_value[0]
                rewards.append(reward)
            # elif gt_emo in top_emotion_list[1]:
            #     reward = top_value[1]
            #     rewards.append(reward)
            else:
                rewards.append(0.0)

    return rewards

def think_top1_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    sub_rewards = []
    videos = kwargs.get("video", "")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    except_list =[]
    # For batch processing
    total_student_think= []
    for i, cont in enumerate(contents):
        try:
            total_student_think.append(re.search(r'<think>(.*?)</think>', cont).group(1).strip())
        except:
            total_student_think.append("")
            except_list.append(i)
    em_inputs = EMOTION_TOKENIZER(total_student_think,
                                  return_tensors="pt",
                                    padding=True,
                                    truncation=True).to(EMOTION_DEVICE)
    with torch.no_grad():
        logits = EMOTION_MODEL(**em_inputs).logits
    probs = F.softmax(logits, dim=1).squeeze()

    topk = torch.topk(probs, k=1)
    top_indices = topk.indices.tolist()
    top_values = topk.values.tolist()

    
    top_labels=[]
    for index_list in top_indices:
        tmp_list = []
        for index in index_list:
            tmp_list.append(EMOTION_MODEL.config.id2label[index])
        top_labels.append(tmp_list) # numgeneration 개수, 각 내부는 topk개씩.
        
    text_emotion_list =[]
    for top_label_list in top_labels:
        tmp_top_label_list = []
        for top_label in top_label_list:
            emotion = match_emotion_EMOBERT(top_label)
            tmp_top_label_list.append(emotion)
        text_emotion_list.append(tmp_top_label_list)
        
    
    ground_truth_emotion_list = []

    # 예측 gt
    for sol in solution:
        # Extract answer from solution if it has think/answer tags
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        ground_truth_emotion_list.append(match_emotion_key_MAFW(ground_truth))


    for i, (top_emotion_list, top_value, gt_emo) in enumerate(zip(text_emotion_list, top_values, ground_truth_emotion_list)):
        if i in except_list:
            rewards.append(0.0)
        else:
            if gt_emo in top_emotion_list:
                reward = 1.0 #top_value[0]
                rewards.append(reward)
            # elif gt_emo in top_emotion_list[1]:
            #     reward = top_value[1]
            #     rewards.append(reward)
            else:
                rewards.append(0.0)

    return rewards

def think_with_hint_reward(completions, solution, **kwargs):
    def compute_keyword_score(text: str, keyword_dict: dict) -> float:
        text_lower = text.lower()  # 대소문자 구분 없이 일치 확인
        total_score = 0.0

        for key, value in keyword_dict.items():
            if key in text_lower:
                total_score += value

        return total_score
    
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    sub_rewards = []
    videos = kwargs.get("video", "")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    except_list =[]
    # For batch processing
    total_student_think= []
    for i, cont in enumerate(contents):
        try:
            total_student_think.append(re.search(r'<think>(.*?)</think>', cont).group(1).strip())
        except:
            total_student_think.append("")
            except_list.append(i)
    em_inputs = EMOTION_TOKENIZER(total_student_think,
                                  return_tensors="pt",
                                    padding=True,
                                    truncation=True).to(EMOTION_DEVICE)
    with torch.no_grad():
        logits = EMOTION_MODEL(**em_inputs).logits
    probs = F.softmax(logits, dim=1).squeeze()
    topk = torch.topk(probs, k=2)
    top_indices = topk.indices.tolist()
    top_values = topk.values.tolist()
    
    top_labels=[]
    for index_list in top_indices:
        tmp_list = []
        for index in index_list:
            tmp_list.append(EMOTION_MODEL.config.id2label[index])
        top_labels.append(tmp_list) # numgeneration 개수, 각 내부는 topk개씩.
        
    text_emotion_list =[]
    for top_label_list in top_labels:
        tmp_top_label_list = []
        for top_label in top_label_list:
            emotion = match_emotion_EMOBERT(top_label)
            tmp_top_label_list.append(emotion)
        text_emotion_list.append(tmp_top_label_list)
        
    
    ground_truth_emotion_list = []

    # 예측 gt
    for sol in solution:
        # Extract answer from solution if it has think/answer tags
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        ground_truth_emotion_list.append(match_emotion_key_MAFW(ground_truth))


    for i, (top_emotion_list, top_value, gt_emo, student_think) in enumerate(zip(text_emotion_list, top_values, ground_truth_emotion_list,total_student_think)):
        if i in except_list:
            rewards.append(0.0)
        else:
            # if gt_emo in top_emotion_list:
            #     reward = 1.0 
            # else:
            #     reward =0.0
                
            # hint 확인.
            key_word_dict = tfidf_dict[gt_emo]
            hint_reward = compute_keyword_score(student_think, key_word_dict)
            hint_reward = min(hint_reward, 1.0)
            # reward = reward + hint_reward
            rewards.append(hint_reward)
        
    return rewards

def think_confusion_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    sub_rewards = []
    videos = kwargs.get("video", "")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    except_list =[]
    # For batch processing
    total_student_think= []
    for i, cont in enumerate(contents):
        try:
            total_student_think.append(re.search(r'<think>(.*?)</think>', cont).group(1).strip())
        except:
            total_student_think.append("")
            except_list.append(i)
    em_inputs = EMOTION_TOKENIZER(total_student_think,
                                  return_tensors="pt",
                                    padding=True,
                                    truncation=True).to(EMOTION_DEVICE)
    with torch.no_grad():
        logits = EMOTION_MODEL(**em_inputs).logits
    probs = F.softmax(logits, dim=1).squeeze()
    topk = torch.topk(probs, k=2)
    top_indices = topk.indices.tolist()
    top_values = topk.values.tolist()
    
    top_labels=[]
    for index_list in top_indices:
        tmp_list = []
        for index in index_list:
            tmp_list.append(EMOTION_MODEL.config.id2label[index])
        top_labels.append(tmp_list) # numgeneration 개수, 각 내부는 topk개씩.
        
    text_emotion_list =[]
    for top_label_list in top_labels:
        tmp_top_label_list = []
        for top_label in top_label_list:
            emotion = match_emotion_EMOBERT(top_label)
            tmp_top_label_list.append(emotion)
        text_emotion_list.append(tmp_top_label_list)
        
    
    ground_truth_emotion_list = []

    # 예측 gt
    for sol in solution:
        # Extract answer from solution if it has think/answer tags
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        ground_truth_emotion_list.append(match_emotion_key_MAFW(ground_truth))


    for i, (top_emotion_list, top_value, gt_emo) in enumerate(zip(text_emotion_list, top_values, ground_truth_emotion_list)):
        contrastive_emotion_list = confusion_matrix_dict[gt_emo]
        if i in except_list:
            rewards.append(0.0)
        else:
            if gt_emo in top_emotion_list:
                reward = 1.0 #top_value[0]
                rewards.append(reward)
            # elif gt_emo in top_emotion_list[1]:
            #     reward = top_value[1]
            #     rewards.append(reward)
            elif top_emotion_list[0] in contrastive_emotion_list:
                reward = -0.5
                rewards.append(reward)
            else:
                rewards.append(0.0)

    return rewards

EMOTION_KEYWORD_DICT = {
    "anger": ["anger", "angry"],
    "sadness": ["sadness","sad", "sorrow"],
    "anxiety": ["anxiety","anxious", "worried"],
    "helplessness": ["helplessness", "helpless","powerless",],
    "disappointment": ["disappointment", "disappointed", "disappoint"],
    "happiness": ["happiness", "happy","joy", "joyful"],
    "contempt": ["contempt", "contemptuous","disdain", "scorn", "disdainful","scornful"],
    "fear": ["fear", "fearful", "scared", "afraid"],
    "surprise": ["surprise", "surprised"],
    "disgust": ["disgust", "disgusted"],
    "neutral": ["neutral"],
}

def think_emotion_label_reward(completions, solution, **kwargs):
    def check_emotion_label(text: str, gt_emotion: str) -> float:
        gt_emotion_keyword_list = EMOTION_KEYWORD_DICT.get(gt_emotion, [])
        text_lower = text.lower()  # 대소문자 구분 없이 일치 확인
        text_lower_list = text_lower.split()
        total_score = 0.0

        for word in text_lower_list:
            if word in gt_emotion_keyword_list:
                total_score += 0.5
        total_score = min(total_score, 1.0)  # Ensure the score does not exceed 1.0
                

        return total_score
    
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    sub_rewards = []
    videos = kwargs.get("video", "")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    except_list =[]
    # For batch processing
    total_student_think= []
    for i, cont in enumerate(contents):
        try:
            total_student_think.append(re.search(r'<think>(.*?)</think>', cont).group(1).strip())
        except:
            total_student_think.append("")
            except_list.append(i)
    em_inputs = EMOTION_TOKENIZER(total_student_think,
                                  return_tensors="pt",
                                    padding=True,
                                    truncation=True).to(EMOTION_DEVICE)
    with torch.no_grad():
        logits = EMOTION_MODEL(**em_inputs).logits
    probs = F.softmax(logits, dim=1).squeeze()
    topk = torch.topk(probs, k=2)
    top_indices = topk.indices.tolist()
    top_values = topk.values.tolist()
    
    top_labels=[]
    for index_list in top_indices:
        tmp_list = []
        for index in index_list:
            tmp_list.append(EMOTION_MODEL.config.id2label[index])
        top_labels.append(tmp_list) # numgeneration 개수, 각 내부는 topk개씩.
        
    text_emotion_list =[]
    for top_label_list in top_labels:
        tmp_top_label_list = []
        for top_label in top_label_list:
            emotion = match_emotion_EMOBERT(top_label)
            tmp_top_label_list.append(emotion)
        text_emotion_list.append(tmp_top_label_list)
        
    
    ground_truth_emotion_list = []

    # 예측 gt
    for sol in solution:
        # Extract answer from solution if it has think/answer tags
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        ground_truth_emotion_list.append(match_emotion_key_MAFW(ground_truth))


    for i, (top_emotion_list, top_value, gt_emo, student_think) in enumerate(zip(text_emotion_list, top_values, ground_truth_emotion_list,total_student_think)):
        if i in except_list:
            rewards.append(0.0)
        else:
            hint_reward = check_emotion_label(student_think, gt_emo)
            rewards.append(hint_reward)
        
    return rewards

def chain_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    videos = kwargs.get("video", "")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    except_list =[]
    # For batch processing
    total_student_think= []
    for i, cont in enumerate(contents):
        try:
            total_student_think.append(re.search(r'<think>(.*?)</think>', cont).group(1).strip())
        except:
            total_student_think.append("")
            except_list.append(i)
    em_inputs = EMOTION_TOKENIZER(total_student_think,
                                  return_tensors="pt",
                                    padding=True,
                                    truncation=True).to(EMOTION_DEVICE)
    with torch.no_grad():
        logits = EMOTION_MODEL(**em_inputs).logits
    probs = F.softmax(logits, dim=1).squeeze()
    topk = torch.topk(probs, k=2)
    top_indices = topk.indices.tolist()
    top_values = topk.values.tolist()
    
    top_labels=[]
    for index_list in top_indices:
        tmp_list = []
        for index in index_list:
            tmp_list.append(EMOTION_MODEL.config.id2label[index])
        top_labels.append(tmp_list) # numgeneration 개수, 각 내부는 topk개씩.
        
    text_emotion_list =[]
    for top_label_list in top_labels:
        tmp_top_label_list = []
        for top_label in top_label_list:
            emotion = match_emotion_EMOBERT(top_label)
            tmp_top_label_list.append(emotion)
        text_emotion_list.append(tmp_top_label_list)
        
    
    ground_truth_emotion_list = []
    student_answer_list = []
    # 예측 gt
    for video, content, sol in zip(videos, contents, solution):
        reward = 0.0
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                ground_truth_emotion_list.append(match_emotion_key_MAFW(ground_truth))
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer_list.append(match_emotion_key_MAFW(student_answer))


            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        
    for i, (top_emotion_list, top_value, gt_emo, student_emo) in enumerate(zip(text_emotion_list, top_values, ground_truth_emotion_list, student_answer_list)):
        if i in except_list:
            rewards.append(0.0)
        else:
            if student_emo ==gt_emo:
                if gt_emo in top_emotion_list:
                    reward = 1.0 #top_value[0]
                    rewards.append(reward)
                # elif gt_emo in top_emotion_list[1]:
                #     reward = top_value[1]
                #     rewards.append(reward)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)

    return rewards


def consistency_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    except_list =[]
    # For batch processing
    total_student_think= []
    
    student_answer_list = []
    
    for i, cont in enumerate(contents):
        try:
            content_match = re.search(r'<answer>(.*?)</answer>', cont)
            student_answer = content_match.group(1).strip() if content_match else cont.strip()
            student_answer_list.append(match_emotion_key_MAFW(student_answer))
            
            total_student_think.append(re.search(r'<think>(.*?)</think>', cont).group(1).strip())
        except:
            total_student_think.append("")
            except_list.append(i)
    em_inputs = EMOTION_TOKENIZER(total_student_think,
                                  return_tensors="pt",
                                    padding=True,
                                    truncation=True).to(EMOTION_DEVICE)
    with torch.no_grad():
        logits = EMOTION_MODEL(**em_inputs).logits
    probs = F.softmax(logits, dim=1).squeeze()
    topk = torch.topk(probs, k=2)
    top_indices = topk.indices.tolist()

    
    top_labels=[]
    for index_list in top_indices:
        tmp_list = []
        for index in index_list:
            tmp_list.append(EMOTION_MODEL.config.id2label[index])
        top_labels.append(tmp_list) # numgeneration 개수, 각 내부는 topk개씩.
        
    text_emotion_list =[]
    for top_label_list in top_labels:
        tmp_top_label_list = []
        for top_label in top_label_list:
            emotion = match_emotion_EMOBERT(top_label)
            tmp_top_label_list.append(emotion)
        text_emotion_list.append(tmp_top_label_list)
        

    for i, (top_emotion_list, student_emo) in enumerate(zip(text_emotion_list, student_answer_list)):
        if i in except_list:
            rewards.append(0.0)
        else:
            if student_emo in top_emotion_list:
                reward = 1.0 
                rewards.append(reward)
            else:
                rewards.append(0.0)

    return rewards


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    videos = kwargs.get("video", "")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for video, content, sol in zip(videos, contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                if 'MAFW' in video:
                    student_answer = match_emotion_key_MAFW(student_answer)
                elif 'DFEW' in video:
                    student_answer = match_emotion_key_DFEW(student_answer)

                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Video: {video}\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def format_reward_2(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "think": think_reward,
    "think_half": think_half_reward,
    "chain": chain_reward,
    "accuracy": accuracy_reward,
    "format": format_reward,
    "consistency": consistency_reward,
    "think_with_hint": think_with_hint_reward,
    "think_confusion": think_confusion_reward,
    "think_emotion_label": think_emotion_label_reward,
    "think_top1": think_top1_reward
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

from datasets import Dataset, DatasetDict
import random


def load_video_dataset(train_json_path: str, test_json_path: str, data_ratio: float = 1.0, test_data_ratio: float = 0.01,shuffle_seed: int = 42):
    #### Train Dataset
    with open(train_json_path, 'r') as file:
        train_data = json.load(file)
        
    rng = random.Random(shuffle_seed)
    rng.shuffle(train_data)
    train_data_length = int(len(train_data) * data_ratio)
    train_data = train_data[:train_data_length]
    # train_data = train_data[train_data_length:]
    transformed_data = {
        'video': [],
        'problem': [],
        'solution': []
    }
    
    for entry in train_data:
        video_path = entry['video']
        problem = None  
        for conversation in entry['conversations']:
            if conversation['from'] == 'human':
                problem = conversation['value'].replace('<video>\n<audio>\n', '')

            elif conversation['from'] == 'gpt' and problem is not None:
                solution = f"<answer> {conversation['value']} </answer>"
                transformed_data['video'].append(video_path)
                transformed_data['problem'].append(problem)
                transformed_data['solution'].append(solution)

    train_dataset = Dataset.from_dict(transformed_data)
    
    ### Test Dataset
    with open(test_json_path, 'r') as file:
        test_data = json.load(file)
    random.shuffle(test_data)
    test_data_length = int(len(test_data) * test_data_ratio)
    test_data = test_data[:test_data_length]
    
    test_transformed_data = {
        'video': [],
        'problem': [],
        'solution': []
    }
    
    for entry in test_data:
        video_path = entry['video']
        problem = None  
        for conversation in entry['conversations']:
            if conversation['from'] == 'human':
                problem = conversation['value'].replace('<video>\n<audio>\n', '')

            elif conversation['from'] == 'gpt' and problem is not None:
                solution = f"<answer> {conversation['value']} </answer>"
                test_transformed_data['video'].append(video_path)
                test_transformed_data['problem'].append(problem)
                test_transformed_data['solution'].append(solution)

    test_dataset = Dataset.from_dict(test_transformed_data)

    dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})
    
    return dataset_dict


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]


    # Load the dataset
    train_json_file_path = script_args.dataset_name 
    test_json_file_path = script_args.test_dataset_name
    data_ratio = script_args.data_ratio
    dataset = load_video_dataset(train_json_file_path, test_json_file_path, data_ratio)
   # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    # def make_conversation_image(example):
    #     return {
    #         "prompt": [
    #             {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image"},
    #                     {"type": "text", "text": example["problem"]},
    #                 ],
    #             },
    #         ],
    #     }


    QUESTION_TEMPLATE = "{Question}\nOutput the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": example["problem"]} #QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])
    elif "video" in dataset[script_args.dataset_train_split].features:
        print("has video in dataset")
        dataset = dataset.map(make_conversation_video)
        
    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    trainer_cls = HumanOmniVLGRPO_TEXT_Trainer 
    print("using: ", trainer_cls)

    # lora로 타겟할 위치 설정
    if script_args.lora_enabled:
        print("lora enabled")

        target_module = []
        for i in range(23):
            target_module.append(f'model.layers.{i}.self_attn.q_proj')
            target_module.append(f'model.layers.{i}.self_attn.k_proj')
            target_module.append(f'model.layers.{i}.self_attn.v_proj')
            target_module.append(f'model.layers.{i}.self_attn.o_proj')

        model_args.lora_target_modules =target_module # ['q_proj', 'k_proj', 'v_proj', 'o_proj']      

    if script_args.bnb_enabled:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_cfg = None
        
        
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        quantization_config=bnb_cfg, 
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        reference_model_switch = script_args.reference_model_switch
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    ### Edit
    # training_args.eval_strategy = script_args.eval_strategy
    main(script_args, training_args, model_args)
