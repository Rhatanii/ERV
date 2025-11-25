import re
import os
import torch
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import match_emotion_key_DFEW, match_emotion_key_MAFW
  

local_rank = int(os.environ.get("LOCAL_RANK", 0))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    EMOTION_DEVICE = torch.device(f"cuda:{local_rank}")
else:
    EMOTION_DEVICE = torch.device("cpu")

_EMOTION_MODEL_ID ="Rhatanii/Emo_Classifier"
EMOTION_TOKENIZER = AutoTokenizer.from_pretrained(_EMOTION_MODEL_ID)
EMOTION_MODEL  = AutoModelForSequenceClassification.from_pretrained(_EMOTION_MODEL_ID)
EMOTION_MODEL.eval()
EMOTION_MODEL.to(EMOTION_DEVICE)

GOEMOTION_TO_LABEL ={
    "happiness": 17,
    "anger": 2,
    "disgust": 11,
    "fear": 14,
    "sadness": 25,
    "surprise": 26,
    "neutral": 27,
    "anxiety": 19,
    "helplessness": 16,
    "disappointment": 9,
    "contempt": 10,
}



def think_step_reward(completions, solution, **kwargs):
    """
        Reward function that calculates the reward based on emotion alignment 
        between the generated thought process and the ground truth emotion.

        Analyzes the thought process (<think>...</think>) text from the generated 
        completions, classifies the emotion of each sentence, and calculates 
        a reward by comparing it to the ground truth emotion
    """
    # 1. Prepare Input Data and Extract Thought Process (<think>) Text
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    except_list =[]

    
    all_sentences = [] # Stores all sentences from all completions.
    sent2example = [] # Stores the number of sentences per completion
    for i, cont in enumerate(contents):
        try:
            # Extract text between <think>...</think> tags.
            think_text = re.search(r'<think>(.*?)</think>', cont, re.DOTALL).group(1).strip()
        except:
            # If <think> tag is not found, treat it as an empty string and add index to exception list.
            think_text = ""
            except_list.append(i)
            
        # Split the text into sentences.
        sentences = re.split(r'(?<=[\.!?])\s+(?=(?:[^"]*"[^"]*")*[^"]*$)', think_text)
        sentences = [s for s in sentences if s]
        for s in sentences:
            all_sentences.append(s)
        sent2example.append(len(sentences)) # Record the number of sentences in this completion.
    
    # 2. Calculate Emotion Logits per Sentence using the ERV
    em_inputs = EMOTION_TOKENIZER(
        all_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(EMOTION_DEVICE)
    
    with torch.no_grad():
        logits = EMOTION_MODEL(**em_inputs).logits
        
    probs = torch.sigmoid(logits) 
    threshold=0.1
    preds = (probs > threshold).int() 
    

    # 3. Prepare Ground Truth Emotion Labels
    ground_truth_emotion_list = []
    ground_truth_emotion_idx_list = []
    for sol in solution:
        # Extract answer from solution 
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        transformed_ground_truth = match_emotion_key_MAFW(ground_truth)
        ground_truth_emotion_list.append(transformed_ground_truth)
        ground_truth_emotion_idx_list.append(GOEMOTION_TO_LABEL[transformed_ground_truth])
    

    # 4. Extract Predicted Emotion Indices per Sentence (Maximum of 2)
    top_indices_list=[]
    for i in range(len(all_sentences)):
        # Get indices of emotions that exceeded the threshold.
        idxs = (preds[i] == 1).nonzero(as_tuple=True)[0].tolist()
        if len(idxs) == 0:
            idxs = [torch.argmax(probs[i]).item()]
        elif len(idxs) > 2:
            idxs = torch.topk(probs[i], k=2).indices.tolist()
        top_indices_list.append(idxs)
    
    # 5. Calculate Statistics per Paragraph: GT Matches and Non-Neutral Count
    sentence_counts = sent2example.copy()
    neutral_idx = GOEMOTION_TO_LABEL["neutral"]
    
    non_neutral_counts = [] # Count of non-neutral sentences per paragraph 
    match_counts = [] # Count of sentences matching the GT emotion per paragraph
    offset = 0

    for para_id, num_sent in enumerate(sentence_counts):
        # Slice the list of predicted emotion indices for sentences belonging to the current paragraph.
        para_preds = top_indices_list[offset : offset + num_sent]

        # 5-1) Calculate Non-neutral Sentence Count
        non_neu = sum(
            not (len(pred) == 1 and pred[0] == neutral_idx)
            for pred in para_preds
        )
        non_neutral_counts.append(non_neu)

        # 5-2) Calculate GT Emotion Match Count
        gt_idx = ground_truth_emotion_idx_list[para_id]
        match_c = sum(gt_idx in pred for pred in para_preds)
        match_counts.append(match_c)

        offset += num_sent
    
    # 6. Calculate Reward
    for match_count, non_neutral_count, total_count, gt_idx in zip(match_counts, non_neutral_counts, sentence_counts,ground_truth_emotion_idx_list):
        if total_count == 0:
            rewards.append(0.0)
        else:
            # 6-1) Calculate the ratio of GT emotion-matching sentences within the paragraph
            if gt_idx == neutral_idx: # Neutral
                reward = match_count / total_count if total_count > 0 else 0.0
            else:
                reward = match_count / non_neutral_count if non_neutral_count > 0 else 0.0
            rewards.append(reward)
    
    return rewards


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    videos = kwargs.get("video", "")

    for video, content, sol in zip(videos, contents, solution):
        reward = 0.0

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
                pass 
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]