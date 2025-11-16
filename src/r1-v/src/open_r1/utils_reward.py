import re
import os
import torch
import json
from datetime import datetime, timezone, timedelta

from math_verify import parse, verify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import read_json_to_dict,  match_emotion_key_DFEW, match_emotion_key_MAFW, match_emotion_EMOBERT


local_rank = int(os.environ.get("LOCAL_RANK", 0))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    EMOTION_DEVICE = torch.device(f"cuda:{local_rank}")
else:
    EMOTION_DEVICE = torch.device("cpu")

_EMOTION_MODEL_ID ="/mnt/ssd_hs/Exp/R1-Omni/src/text_emotion_classifier/emotion_trainer-TOTAL-EMER-MERR-gpt_ver_label_28_multi-balance-final-epoch10-bs64-ga1-lr3e-05-tr0.5/checkpoint-460"
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

gpt_keyword_path = "/mnt/ssd_hs/Dataset/R1-Omni/clues/emotion_clues.json"
emotion_keyword_dict = read_json_to_dict(gpt_keyword_path)
confusion_matrix_path = "/mnt/ssd_hs/Dataset/R1-Omni/clues/emotion_confusion_matrix_max.json"
confusion_matrix_dict = read_json_to_dict(confusion_matrix_path)
TF_IDF_PATH= "/mnt/ssd_hs/Dataset/R1-Omni/clues/tfidf_scores.json"
tfidf_dict = read_json_to_dict(TF_IDF_PATH)


######################################
# modality model
_MODALITY_MODEL_ID="/mnt/ssd_hs/Exp/R1-Omni/src/text_modality_classifier/deberta_Modality-optimizer-3cls-epoch10-bs32-ga2-lr0.0001/checkpoint-573"
MODALITY_TOKENIZER= AutoTokenizer.from_pretrained(_MODALITY_MODEL_ID)
MODALITY_MODEL  = AutoModelForSequenceClassification.from_pretrained(_MODALITY_MODEL_ID)
MODALITY_MODEL.eval()
MODALITY_MODEL.to(EMOTION_DEVICE)
######################################

# model tokenizer
MODEL_NAME="/mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0525-EMER-MERR-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower/checkpoint-150"
MODEL_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

###########################################
import csv
def load_samples_as_dict(csv_path: str, dataset_name: str)-> dict:
    """
    주어진 CSV 파일을 읽고 sample_id를 key로, {video, audio, video_audio}를 value로 하는 dict를 반환합니다.
    """
    result = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row['sample_id'])
            result[sample_id] = {
                'video': int(row['video']),
                'audio': int(row['audio']),
                'video_audio': int(row['video_audio'])
            }
    return result
DFEW_important_modality_path = "/mnt/ssd_hs/Exp/R1-Omni/analysis/DFEW_modality_results_train.csv"
MAFW_important_modality_path = "/mnt/ssd_hs/Exp/R1-Omni/analysis/MAFW_modality_results_train.csv"
DFEW_important_modality_dict = load_samples_as_dict(DFEW_important_modality_path, "DFEW")
MAFW_important_modality_dict = load_samples_as_dict(MAFW_important_modality_path, "MAFW")

def av_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format depending on modality."""
    videos = kwargs.get("video", "")
    important_modality_list = []
    
    for video in videos:
        video_id = video.split('/')[-1].split('.')[0]
        if 'MAFW' in video:
            important_modality_dict = MAFW_important_modality_dict
        elif 'DFEW' in video:
            important_modality_dict = DFEW_important_modality_dict
        modality_dict = important_modality_dict.get(str(video_id), {})
        
        video_flag = modality_dict.get('video', 0)
        audio_flag = modality_dict.get('audio', 0)
        if video_flag == 1 and audio_flag == 0:
            important_modality_list.append('vision')
        elif video_flag == 0 and audio_flag == 1:
            important_modality_list.append('audio')
        else:
            important_modality_list.append('multimodal')
            

    vision_pattern = (
        r"^\s*<vis_desc>.*?</vis_desc>\s*"
        r"<aud_desc>.*?</aud_desc>\s*"
        r"<think>.*?</think>\s*"
        r"<answer>.*?</answer>\s*$"
    )
    audio_pattern = (
        r"^\s*<aud_desc>.*?</aud_desc>\s*"
        r"<vis_desc>.*?</vis_desc>\s*"
        r"<think>.*?</think>\s*"
        r"<answer>.*?</answer>\s*$"
    )
    rewards = []
    completion_contents = [completion[0]["content"] for completion in completions]
    for idx, content in enumerate(completion_contents):
        modality = important_modality_list[idx]

        if modality == 'vision':
            match = re.fullmatch(vision_pattern, content, re.DOTALL)
            rewards.append(1.0 if match else 0.0)

        elif modality == 'audio':
            match = re.fullmatch(audio_pattern, content, re.DOTALL)
            rewards.append(1.0 if match else 0.0)
        else:  # multimodal
            match = (
                re.fullmatch(vision_pattern, content, re.DOTALL)
                or re.fullmatch(audio_pattern, content, re.DOTALL)
            )
            rewards.append(0.5 if match else -1.0)




    return rewards




from itertools import accumulate
from collections import defaultdict


import ahocorasick


def build_emotion_automaton(keywords):
    A = ahocorasick.Automaton()
    for kw in keywords:
        A.add_word(kw.lower(), kw)
    A.make_automaton()
    return A

# Emotion Keyword path
EMO_KEYWORD_PATH="/mnt/ssd_hs/Exp/R1-Omni/Interactive_MMReasoning/dataset/emotional_keywords_accumulated.json"
with open(EMO_KEYWORD_PATH, "r", encoding="utf-8") as f:
    EMOTIONAL_KEYWORDS_DICT = json.load(f)
emotional_keywords = set(EMOTIONAL_KEYWORDS_DICT["emotional_keywords"])
EMOTION_AUTOMATON = build_emotion_automaton(emotional_keywords)

def find_keywords_in_sentence(sentence, automaton):
    """Return list of (matched_keyword, end_index) in sentence.
       Only returns matches that are *whole words* (respecting word boundaries)."""
    results = []
    s = sentence.lower()
    length = len(s)

    for end_idx, kw in automaton.iter(s):
        start_idx = end_idx - len(kw) + 1

        # 문자 경계 판별
        left_ok = start_idx == 0 or not s[start_idx - 1].isalpha()
        right_ok = end_idx == length - 1 or not s[end_idx + 1].isalpha()

        if left_ok and right_ok:
            results.append((kw, end_idx))
    
    return results


def map_sentences_to_token_indices(sentences_with_idx, tokenizer):
    """
    sentences_with_idx: list of tuples (sentence, batch_idx)
    tokenizer: model tokenizer
    return: list of tuples (batch_idx, start_idx, end_idx) for each sentence
    """
    # batch별 문장 리스트 구성
    batch_to_sentences = defaultdict(list)
    for sent, batch_idx in sentences_with_idx:
        batch_to_sentences[batch_idx].append(sent)

    batch_sent_spans = []

    for batch_idx, sents in batch_to_sentences.items():
        # 각 문장의 토큰 길이
        token_lengths = [len(tokenizer(sent, add_special_tokens=False).input_ids) for sent in sents]
        cum_lengths = list(accumulate(token_lengths))

        prev_end = 0
        for end in cum_lengths:
            batch_sent_spans.append((batch_idx, prev_end, end))
            prev_end = end

    return batch_sent_spans




import re, json, bisect
from collections import defaultdict
import torch

# --- Util: 문장을 span과 함께 분리 (인용부호 내 구두점 예외는 기존 정규식 유지) ---
_SENT_SPLIT_RE = re.compile(r'(?<=[\.!?])\s+(?=(?:[^"]*"[^"]*")*[^"]*$)')

def split_sentences_with_spans(text: str):
    """Return list of (sent_text, char_start, char_end) within `text`."""
    spans = []
    start = 0
    for m in _SENT_SPLIT_RE.finditer(text):
        end = m.start()
        seg = text[start:end]
        if seg:
            spans.append((seg, start, end))
        start = m.end()
    # tail
    if start < len(text):
        spans.append((text[start:], start, len(text)))
    return spans

# --- Util: 단어 경계 판단 (유니코드 글자 기준) ---
def _is_letter(ch: str) -> bool:
    return ch.isalpha()  # 한글 포함 True

def _is_word_boundary(s: str, idx: int) -> bool:
    """idx는 문자 '사이' 위치: 0<=idx<=len(s). 좌/우가 글자가 아니면 경계."""
    if idx <= 0 or idx >= len(s): 
        return True
    return (not _is_letter(s[idx-1])) or (not _is_letter(s[idx]))

# --- 감정 키워드 탐지(문단 단위, 경계 체크) ---
def find_keywords_in_text(text: str, automaton):
    """
    Return list of dicts: {"kw": str, "char_start": int, "char_end": int}
    경계는 단어 경계 기반.
    """
    results = []
    s_lower = text.lower()
    for end_idx, kw in automaton.iter(s_lower):
        kw_lower = kw.lower()
        start_idx = end_idx - len(kw_lower) + 1
        # 경계 검사 (원문 길이와 lower 길이는 동일하다고 가정: 일반 라틴/한글 OK)
        if _is_word_boundary(s_lower, start_idx) and _is_word_boundary(s_lower, end_idx + 1):
            results.append({"kw": kw, "char_start": start_idx, "char_end": end_idx + 1})
    return results

# --- 토크나이저 캐시: 문단 단위 토큰/오프셋 ---
def build_doc_token_cache(contents, tokenizer):
    """
    Return list per batch:
      {
        "text": str,
        "input_ids": List[int],
        "offsets": List[Tuple[start,end]],  # add_special_tokens=False
      }
    """
    cache = []
    for t in contents:
        enc = tokenizer(
            t,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False
        )
        # HF fast tokenizer는 offset_mapping을 리스트로 반환
        offsets = enc["offset_mapping"]
        cache.append({"text": t, "input_ids": enc["input_ids"], "offsets": offsets})
    return cache

# --- 문자 오프셋 -> 토큰 인덱스 (시작 토큰) ---
def char_to_token_start(offsets, char_start):
    """
    offsets: List[(s,e)]
    char_start: 키워드 시작 문자 위치
    return: 키워드가 시작되는 첫 토큰 인덱스(없으면 None)
    """
    # 첫 토큰 whose offset_start >= char_start (또는 char_start in [start,end))
    # 보통은 'start <= char_start < end'로 포함되는 토큰을 고르는 게 자연스럽습니다.
    # 포함 토큰이 없다면, 다음 토큰의 start >= char_start인 것을 선택.
    # 이진탐색 사용
    starts = [st for st, _ in offsets]
    idx = bisect.bisect_left(starts, char_start)
    # 보정: char_start가 이전 토큰 범위 내부면 그 토큰으로
    if idx > 0:
        st, en = offsets[idx-1]
        if st <= char_start < en:
            return idx-1
    # 아니라면 idx가 후보 (단, idx==len이면 없음)
    if idx < len(offsets):
        return idx
    return None

# --- 문장 span -> 토큰 span 맵핑 ---
def span_chars_to_token_span(offsets, char_start, char_end):
    """
    char_start/end 포함 범위를 덮는 토큰 구간 [tok_s, tok_e)
    """
    # 시작 토큰
    tok_s = char_to_token_start(offsets, char_start)
    if tok_s is None:
        return None
    # 끝 토큰: char_end-1을 커버하는 토큰의 다음 인덱스
    last_char = max(char_start, char_end - 1)
    tok_e = char_to_token_start(offsets, last_char)
    if tok_e is None:
        tok_e = len(offsets) - 1
    # tok_e는 last_char를 포함하는 토큰 인덱스 → 구간 끝은 +1
    return tok_s, min(tok_e + 1, len(offsets))



# --- 메인: reward ---
def think_emotion_keyword_attention_reward(completions, solution, **kwargs):
    """
    Reward using modality-specific attention around emotional keywords.
    Assumptions:
      - attention_*_scores shape: (B, answer, L)
      - Query dim = generated tokens within the 'answer' window (e.g., reasoning tokens)
      - Key dim = full context (prompt + modalities + generated)
    """
    # -------------------------
    # Load contents
    # -------------------------
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    except_list = []
    all_sentences = []             # 텍스트만
    all_sentence_meta = []         # (batch_idx, sent_char_start, sent_char_end)
    think_texts = []               # 배치별 문단(<think> 내부)

    # 임계값 
    THRESH_ABS = 0.7
    THRESH_REL = 1.50

    # -------------------------
    # 1) <think> 텍스트 추출 + 문장 분할(문자 span 포함)
    # -------------------------
    per_batch_sentence_spans = []  # List[List[(sent, s, e)]]
    for i, cont in enumerate(contents):
        try:
            think_text = re.search(r'<think>(.*?)</think>', cont, re.DOTALL).group(1)
        except:
            think_text = ""
            except_list.append(i)

        # think_texts.append(think_text)
        # sent_spans = split_sentences_with_spans(think_text) # [(sentence,  character start, character end)]
        cont_spans = split_sentences_with_spans(cont)
        per_batch_sentence_spans.append(cont_spans)

        for (s, cs, ce) in cont_spans:
            all_sentences.append(s)
            all_sentence_meta.append((i, cs, ce))

    # -------------------------
    # 2) 문장 → 모달리티 분류 (0=vision, 1=audio, 2=none)
    # -------------------------
    modality_inputs = MODALITY_TOKENIZER(
        all_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(EMOTION_DEVICE)
    
    with torch.no_grad():
        preds = MODALITY_MODEL(**modality_inputs)
    modality_indices = torch.argmax(preds.logits, dim=-1).tolist()

    # -------------------------
    # 3) 어텐션 텐서
    # -------------------------
    vision_attn = kwargs.get("attention_vision_scores", None)  # (B, answer) 
    audio_attn  = kwargs.get("attention_audio_scores",  None) # (B, answer) 

    B, Q = vision_attn.shape # Q: Answer 길이

    # 쿼리 시작 옵셋 기본값
    query_starts_in_answer=None
    if query_starts_in_answer is None:
        query_starts_in_answer = [0] * B


    # -------------------------
    # 4) 문단 단위 토크나이즈 + offset_mapping
    # -------------------------
    doc_cache = build_doc_token_cache(contents, MODEL_TOKENIZER) #[{'text': 문단, 'input_ids': [], 'offsets': []}, ...]

    # 문장별 토큰 스팬(쿼리 축에서 필요: 문단 토큰 인덱스 = 쿼리 로컬 인덱스라고 가정)
    # 문장 스팬은 문단 오프셋으로부터 tok_s,tok_e로 변환
    per_batch_sentence_token_spans = []  # List[List[(tok_s, tok_e)]]
    for b in range(B):
        offsets = doc_cache[b]["offsets"]
        sent_tok_spans = []
        for sent, cs, ce in per_batch_sentence_spans[b]:
            tes = span_chars_to_token_span(offsets, cs, ce)
            if tes is None:
                sent_tok_spans.append((None, None))
            else:
                sent_tok_spans.append(tes)
        per_batch_sentence_token_spans.append(sent_tok_spans)

    # -------------------------
    # 5) 문단 단위로 감정 키워드 탐지 → 각 문장에 속하는 매치만 사용
    # -------------------------
    batch_counts = defaultdict(lambda: {"total": 0, "matched": 0})

    # all_sentences / all_sentence_meta / modality_indices 는 같은 길이
    idx_global = 0
    for b in range(B):
        text_b = doc_cache[b]["text"]
        offsets_b = doc_cache[b]["offsets"]

        # 문단 전체에서 키워드 매치 (문자 오프셋)
        kw_matches = find_keywords_in_text(text_b, EMOTION_AUTOMATON)

        # 문장 루프
        for j, (sentence, cs, ce) in enumerate(per_batch_sentence_spans[b]):
            sent_modality = modality_indices[idx_global]
            idx_global += 1

            tok_span = per_batch_sentence_token_spans[b][j]
            if tok_span == (None, None):
                continue
            sent_tok_s, sent_tok_e = tok_span  # 문단 토큰 기준 [s,e)

            # 문장 내 키워드만 추림
            sent_kw_matches = [m for m in kw_matches if (m["char_start"] >= cs and m["char_end"] <= ce)]
            if not sent_kw_matches:
                continue

            # 어텐션 텐서 선택 (쿼리/키 모두에서 사용할 원본)
            # 목적: 쿼리 시점 = "키워드 시작 토큰의 직전" q_idx,
            # 해당 q에서 모달리티 키 범위를 얼마나 봤는지 측정


            for m in sent_kw_matches:
                batch_counts[b]["total"] += 1
                matched = False
                
                # 키워드 시작 토큰 인덱스(문단 로컬)
                kw_tok = char_to_token_start(offsets_b, m["char_start"])
                if kw_tok is None:
                    continue
                # 쿼리 인덱스(q_idx): 키워드 직전 토큰
                q_local = max(sent_tok_s, kw_tok - 1)

                # 쿼리 로컬 -> 쿼리 절대(answer 차원) 변환
                q_abs = query_starts_in_answer[b] + q_local
                if not (0 <= q_abs < Q):
                    # 범위 밖이면 skip
                    continue

                # 해당 쿼리에서 키 축 분포
                v_row = vision_attn[b, q_abs]   # shape: (L,)
                a_row = audio_attn[b, q_abs]   # shape: (L,)

                # 문장 텍스트의 키 범위: 모델의 키 축은 전체 L(프롬프트+모달리티+생성)
                # 문단 토큰은 answer 영역의 일부라고 가정 → 키 축에서의 시작 옵셋 = prompt_lengths[b]
                key_sent_s = query_starts_in_answer[b] + sent_tok_s
                key_sent_e = query_starts_in_answer[b] + sent_tok_e
                key_sent_s = max(0, min(Q, key_sent_s))
                key_sent_e = max(0, min(Q, key_sent_e))
                if key_sent_e <= key_sent_s:
                    continue


                # 문장 내부의 평균 주의(베이스라인)
                v_base = vision_attn[b, key_sent_s:key_sent_e].mean().item()
                a_base = audio_attn[b, key_sent_s:key_sent_e].mean().item()

                # 판정
                if sent_modality == 0:  # vision
                    if (v_row >= THRESH_REL * v_base): #(v_row >= THRESH_ABS):
                        matched = True

                elif sent_modality == 1:  # audio
                    if (a_row >= THRESH_REL * a_base): #(a_row >= THRESH_ABS):
                        matched = True

                else:  # none → 둘 다 일정 수준 이상이면 OK (strict)
                    if ((v_row >= THRESH_REL * v_base) and
                        (a_row >= THRESH_REL * a_base)):
                        matched = True


                if matched:
                    batch_counts[b]["matched"] += 1

    # -------------------------
    # 6) 집계
    # -------------------------
    for i in range(len(completions)):
        total = batch_counts[i]["total"]
        matched = batch_counts[i]["matched"]
        rewards.append(0.0 if total == 0 else matched / total)
        
    log_path = os.getenv("LOG_PATH")
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    KST = timezone(timedelta(hours=9))

    # 현재 시간을 KST로 변환해 문자열로 포맷
    current_time = datetime.now(KST).strftime("%d-%H-%M-%S-%f")
    if local_rank ==0:
        total_count = sum(batch_counts[i]["total"] for i in range(len(completions)))
        matched_count = sum(batch_counts[i]["matched"] for i in range(len(completions)))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"------------- {current_time} Emotional Keywords reward: {sum(rewards)/len(rewards)} -------------\n")
            f.write(f"Total counts: {total_count}\n")
            f.write(f"matched_count: {matched_count}\n")
            f.write(f"Content: {contents[0]}\n")
    return rewards



def think_step_av_with_neutral_reward(completions, solution, **kwargs):
    """Reward function that evaluates emotion from specific sections depending on important modality."""
    contents = [completion[0]["content"] for completion in completions]
    videos = kwargs.get("video", [])

    rewards = []
    except_list = []
    all_sentences = []
    sent2example = []

    # --- 1. important_modality_list 구성 ---
    important_modality_list = []
    for video in videos:
        video_id = video.split('/')[-1].split('.')[0]
        if 'MAFW' in video:
            important_modality_dict = MAFW_important_modality_dict
        elif 'DFEW' in video:
            important_modality_dict = DFEW_important_modality_dict
        else:
            important_modality_dict = {}

        modality_dict = important_modality_dict.get(str(video_id), {})
        video_flag = modality_dict.get('video', 0)
        audio_flag = modality_dict.get('audio', 0)

        if video_flag == 1 and audio_flag == 0:
            important_modality_list.append('vision')
        elif video_flag == 0 and audio_flag == 1:
            important_modality_list.append('audio')
        else:
            important_modality_list.append('multimodal')

    # --- 2. 문장 추출 ---
    for i, cont in enumerate(contents):
        modality = important_modality_list[i]

        sections = []
        if modality == 'audio':
            # 오디오 중심: aud_desc + think
            for tag in ["aud_desc", "think"]:
                match = re.search(fr"<{tag}>(.*?)</{tag}>", cont, re.DOTALL)
                if match:
                    sections.append(match.group(1).strip())

        elif modality == 'vision':
            # 비전 중심: vis_desc + think
            for tag in ["vis_desc", "think"]:
                match = re.search(fr"<{tag}>(.*?)</{tag}>", cont, re.DOTALL)
                if match:
                    sections.append(match.group(1).strip())

        else:
            # 멀티모달: aud_desc + vis_desc + think
            for tag in ["vis_desc", "aud_desc", "think"]:
                match = re.search(fr"<{tag}>(.*?)</{tag}>", cont, re.DOTALL)
                if match:
                    sections.append(match.group(1).strip())

        combined_text = "\n".join(sections)

        if not combined_text:
            except_list.append(i)
            sentences = []
        else:
            # 문장 단위 분리
            sentences = re.split(
                r'(?<=[\.!?])\s+(?=(?:[^"]*"[^"]*")*[^"]*$)',
                combined_text
            )
            sentences = [s.strip() for s in sentences if s.strip()]

        all_sentences.extend(sentences)
        sent2example.append(len(sentences))

    # --- 3. 감정 모델 예측 ---
    em_inputs = EMOTION_TOKENIZER(
        all_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(EMOTION_DEVICE)

    with torch.no_grad():
        logits = EMOTION_MODEL(**em_inputs).logits

    probs = torch.sigmoid(logits)
    threshold = 0.1
    preds = (probs > threshold).int()

    # --- 4. GT 감정 인덱스 준비 ---
    ground_truth_emotion_list = []
    ground_truth_emotion_idx_list = []
    for sol in solution:
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        transformed_ground_truth = match_emotion_key_MAFW(ground_truth)
        ground_truth_emotion_list.append(transformed_ground_truth)
        ground_truth_emotion_idx_list.append(GOEMOTION_TO_LABEL[transformed_ground_truth])

    # --- 5. 문장별 예측 top index ---
    top_indices_list = []
    for i in range(len(all_sentences)):
        idxs = (preds[i] == 1).nonzero(as_tuple=True)[0].tolist()
        if len(idxs) == 0:
            idxs = [torch.argmax(probs[i]).item()]
        elif len(idxs) > 2:
            idxs = torch.topk(probs[i], k=2).indices.tolist()
        top_indices_list.append(idxs)

    sentence_counts = sent2example.copy()
    neutral_idx = GOEMOTION_TO_LABEL["neutral"]

    non_neutral_counts = []
    match_counts = []
    offset = 0

    # --- 6. Reward 계산 ---
    for para_id, num_sent in enumerate(sentence_counts):
        para_preds = top_indices_list[offset:offset + num_sent]

        non_neu = sum(
            not (len(pred) == 1 and pred[0] == neutral_idx)
            for pred in para_preds
        )
        non_neutral_counts.append(non_neu)

        gt_idx = ground_truth_emotion_idx_list[para_id]
        match_c = sum(gt_idx in pred for pred in para_preds)
        match_counts.append(match_c)

        offset += num_sent

    for match_count, non_neutral_count, total_count, gt_idx in zip(
        match_counts, non_neutral_counts, sentence_counts, ground_truth_emotion_idx_list
    ):
        if total_count == 0:
            rewards.append(0.0)
        else:
            reward = match_count / total_count
            rewards.append(reward)

    return rewards


def think_step_with_neutral_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    except_list =[]
    # For batch processing
    total_student_think= []
    
    all_sentences = []
    sent2example = []
    for i, cont in enumerate(contents):
        try:
            think_text = re.search(r'<think>(.*?)</think>', cont, re.DOTALL).group(1).strip()
        except:
            think_text = ""
            except_list.append(i)
        sentences = re.split(r'(?<=[\.!?])\s+(?=(?:[^"]*"[^"]*")*[^"]*$)', think_text)
        sentences = [s for s in sentences if s]
        for s in sentences:
            all_sentences.append(s)
        sent2example.append(len(sentences))

    em_inputs = EMOTION_TOKENIZER(
        all_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(EMOTION_DEVICE)
    
    with torch.no_grad():
        logits = EMOTION_MODEL(**em_inputs).logits
        
    probs = torch.sigmoid(logits)    # logits의 차원은 (numgeneration, num_emotion) 
    threshold=0.1
    preds = (probs > threshold).int() 
    

    # 3) GT emotion 인덱스 준비 (예시: solution이 문자열 리스트라면 매핑 필요)
    ground_truth_emotion_list = []
    ground_truth_emotion_idx_list = []
    for sol in solution:
        # Extract answer from solution if it has think/answer tags
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        transformed_ground_truth = match_emotion_key_MAFW(ground_truth)
        ground_truth_emotion_list.append(transformed_ground_truth)
        ground_truth_emotion_idx_list.append(GOEMOTION_TO_LABEL[transformed_ground_truth])
    

    top_indices_list=[]
    for i in range(len(all_sentences)):
        idxs = (preds[i] == 1).nonzero(as_tuple=True)[0].tolist()
        if len(idxs) == 0:
            idxs = [torch.argmax(probs[i]).item()]
        elif len(idxs) > 2:
            idxs = torch.topk(probs[i], k=2).indices.tolist()
        top_indices_list.append(idxs)
    
    sentence_counts = sent2example.copy()
    neutral_idx = GOEMOTION_TO_LABEL["neutral"]
    
    non_neutral_counts = []
    match_counts = []
    offset = 0

    for para_id, num_sent in enumerate(sentence_counts):
        # 이 문단에 속한 문장들의 top_indices slice
        para_preds = top_indices_list[offset : offset + num_sent]

        # 2-1) neutral이 아닌 문장 개수
        non_neu = sum(
            not (len(pred) == 1 and pred[0] == neutral_idx)
            for pred in para_preds
        )
        non_neutral_counts.append(non_neu)

        # 2-2) GT emotion 일치 문장 개수
        gt_idx = ground_truth_emotion_idx_list[para_id]
        match_c = sum(gt_idx in pred for pred in para_preds)
        match_counts.append(match_c)

        offset += num_sent
    
    for match_count, non_neutral_count, total_count, gt_idx in zip(match_counts, non_neutral_counts, sentence_counts,ground_truth_emotion_idx_list):
        if total_count == 0:
            rewards.append(0.0)
        else:
            # 3) 문단 내에서 GT emotion이 일치하는 문장 비율 계산
            reward = match_count / total_count if total_count > 0 else 0.0
            rewards.append(reward)
    

    return rewards


def think_av_attention_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    except_list =[]
    # For batch processing
    total_student_think= []
    
    all_sentences = []
    all_sentences_with_idx = []
    for i, cont in enumerate(contents):
        try:
            think_text = re.search(r'<think>(.*?)</think>', cont, re.DOTALL).group(1).strip()
        except:
            think_text = ""
            except_list.append(i)
        sentences = re.split(r'(?<=[\.!?])\s+(?=(?:[^"]*"[^"]*")*[^"]*$)', think_text)
        sentences = [s for s in sentences if s]
        for s in sentences:
            all_sentences.append(s)
            all_sentences_with_idx.append((s,i)) # (sentence, example idx)

    # Modality Tokenizer
    modality_inputs = MODALITY_TOKENIZER(
        all_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(EMOTION_DEVICE)
    
    # Modality Verifier / preds에 대한 검증 필요.
    with torch.no_grad():
        preds = MODALITY_MODEL(**modality_inputs)
    modality_indices = torch.argmax(preds.logits, dim=-1).tolist()
    

    # Attn 정리
    # prompt_attn_scores = kwargs.get("attention_prompt_scores",None)
    vision_attn_scores = kwargs.get("attention_vision_scores",None)
    audio_attn_scores = kwargs.get("attention_audio_scores",None)
    reason_attn_scores = kwargs.get("attention_reason_scores",None)
    
    # Sentence 별 index 분리.

    batch_sent_token_spans = map_sentences_to_token_indices(all_sentences_with_idx, MODEL_TOKENIZER)
    
    # Attn 부여
    sentence_attn_scores = []
    for (batch_idx, start, end) in batch_sent_token_spans:
        # 문장 구간에 해당하는 토큰들에 대한 attention 평균
        sent_visual_score = vision_attn_scores[batch_idx, ..., start:end].mean().item()
        sent_audio_score = audio_attn_scores[batch_idx, ..., start:end].mean().item()
        sent_reason_score = reason_attn_scores[batch_idx, ..., start:end].mean().item()
        sentence_attn_scores.append((batch_idx, (sent_visual_score, sent_audio_score, sent_reason_score)))


    # Reward 제공.
    batch_counts = defaultdict(dict)

    for i, (modality_index, (batch_idx, (vis_attn_score, aud_attn_score, reason_attn_score))) in enumerate(zip(modality_indices, sentence_attn_scores)):

        batch_counts[batch_idx]['total_count'] = batch_counts[batch_idx].get('total_count', 0) + 1
        if modality_index == 0:
            if vis_attn_score > aud_attn_score:
                batch_counts[batch_idx]['match_count'] = batch_counts[batch_idx].get('match_count', 0) + 1
        elif modality_index == 1:
            if aud_attn_score > vis_attn_score:
                batch_counts[batch_idx]['match_count'] = batch_counts[batch_idx].get('match_count', 0) + 1

    for i in batch_counts.keys():
        total_count = batch_counts[i].get('total_count', 0)
        match_count = batch_counts[i].get('match_count', 0)
        if total_count == 0:
            rewards.append(0.0)
        else:
            reward = match_count / total_count if total_count > 0 else 0.0
            rewards.append(reward)

    

    return rewards

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
    probs = torch.sigmoid(logits)    # logits의 차원은 (numgeneration, num_emotion) 
    B, C = probs.shape
    
    threshold=0.5
    preds = (probs > threshold).int() 
    
    top_indices_list=[]
    for i in range(B):
        idxs = (preds[i] == 1).nonzero(as_tuple=True)[0].tolist()
        if len(idxs) == 0:
            idxs = [torch.argmax(probs[i]).item()]
        elif len(idxs) > 2:
            idxs = torch.topk(probs[i], k=2).indices.tolist()

        top_indices_list.append(idxs)
        
    
    ground_truth_emotion_list = []
    for sol in solution:
        # Extract answer from solution if it has think/answer tags
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        ground_truth_emotion_list.append(match_emotion_key_MAFW(ground_truth))


    for i, (top_idx, gt_emo) in enumerate(zip(top_indices_list, ground_truth_emotion_list)):
        if i in except_list:
            rewards.append(0.0)
        else:
            reward=0.0
            gt_label = GOEMOTION_TO_LABEL[gt_emo] # 확인하고자하는 label
            if gt_label in top_idx:
                reward =1.0
            rewards.append(reward)

    return rewards

def think_step_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    except_list =[]
    # For batch processing
    total_student_think= []
    
    all_sentences = []
    sent2example = []
    for i, cont in enumerate(contents):
        try:
            think_text = re.search(r'<think>(.*?)</think>', cont, re.DOTALL).group(1).strip()
        except:
            think_text = ""
            except_list.append(i)
        sentences = re.split(r'(?<=[\.!?])\s+(?=(?:[^"]*"[^"]*")*[^"]*$)', think_text)
        sentences = [s for s in sentences if s]
        for s in sentences:
            all_sentences.append(s)
        sent2example.append(len(sentences))

    em_inputs = EMOTION_TOKENIZER(
        all_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(EMOTION_DEVICE)
    
    with torch.no_grad():
        logits = EMOTION_MODEL(**em_inputs).logits
        
    probs = torch.sigmoid(logits)    # logits의 차원은 (numgeneration, num_emotion) 
    threshold=0.1
    preds = (probs > threshold).int() 
    

    # 3) GT emotion 인덱스 준비 (예시: solution이 문자열 리스트라면 매핑 필요)
    ground_truth_emotion_list = []
    ground_truth_emotion_idx_list = []
    for sol in solution:
        # Extract answer from solution if it has think/answer tags
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        transformed_ground_truth = match_emotion_key_MAFW(ground_truth)
        ground_truth_emotion_list.append(transformed_ground_truth)
        ground_truth_emotion_idx_list.append(GOEMOTION_TO_LABEL[transformed_ground_truth])
    

    top_indices_list=[]
    for i in range(len(all_sentences)):
        idxs = (preds[i] == 1).nonzero(as_tuple=True)[0].tolist()
        if len(idxs) == 0:
            idxs = [torch.argmax(probs[i]).item()]
        elif len(idxs) > 2:
            idxs = torch.topk(probs[i], k=2).indices.tolist()
        top_indices_list.append(idxs)
    
    sentence_counts = sent2example.copy()
    neutral_idx = GOEMOTION_TO_LABEL["neutral"]
    
    non_neutral_counts = []
    match_counts = []
    offset = 0

    for para_id, num_sent in enumerate(sentence_counts):
        # 이 문단에 속한 문장들의 top_indices slice
        para_preds = top_indices_list[offset : offset + num_sent]

        # 2-1) neutral이 아닌 문장 개수
        non_neu = sum(
            not (len(pred) == 1 and pred[0] == neutral_idx)
            for pred in para_preds
        )
        non_neutral_counts.append(non_neu)

        # 2-2) GT emotion 일치 문장 개수
        gt_idx = ground_truth_emotion_idx_list[para_id]
        match_c = sum(gt_idx in pred for pred in para_preds)
        match_counts.append(match_c)

        offset += num_sent
    
    for match_count, non_neutral_count, total_count, gt_idx in zip(match_counts, non_neutral_counts, sentence_counts,ground_truth_emotion_idx_list):
        if total_count == 0:
            rewards.append(0.0)
        else:
            # 3) 문단 내에서 GT emotion이 일치하는 문장 비율 계산
            if gt_idx == neutral_idx: # Neutral
                reward = match_count / total_count if total_count > 0 else 0.0
            else:
                reward = match_count / non_neutral_count if non_neutral_count > 0 else 0.0
            rewards.append(reward)
    

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
