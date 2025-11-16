import os
import copy
import warnings
import shutil
from functools import partial

import torch
import torch.nn.functional as F
from .model import load_pretrained_model
from .mm_utils import process_image, process_video, process_audio,tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria,process_image_npary
from .constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP, DEFAULT_AUDIO_TOKEN
import transformers
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, BitsAndBytesConfig


local_rank = int(os.environ.get("LOCAL_RANK", 0))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    EMOTION_DEVICE = torch.device(f"cuda:{local_rank}")
else:
    EMOTION_DEVICE = torch.device("cpu")



_EMOTION_MODEL_ID ="/mnt/ssd_hs/Exp/R1-Omni/src/text_emotion_classifier/emotion_trainer-TOTAL-EMER-MERR-gpt_ver_label_28_multi-balance-final-epoch10-bs64-ga1-lr3e-05-tr0.5/checkpoint-460"
# "/mnt/ssd_hs/Exp/R1-Omni/src/text_emotion_classifier/emotion_trainer-MAFW-EMER-MERR-optimizer-datadata/total_merged_-epoch10-bs64-lr0.0002/checkpoint-168"
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
REVERSE_GOEMOTION_TO_LABEL = {v: k for k, v in GOEMOTION_TO_LABEL.items()}

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


def model_init(model_path=None, **kwargs):
    # with_face = kwargs.get('with_face', False)
    model_path = "HumanOmni_7B" if model_path is None else model_path
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, processor, context_len, audio_processor = load_pretrained_model(model_path, None, model_name, **kwargs)

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES
    if "qwen2vit" in model_path:
        from .mm_utils import process_image_qwen, process_video_qwen
        processor = {
            'image': partial(process_image_qwen, processor=processor, aspect_ratio=None),
            'video': partial(process_video_qwen, processor=processor, aspect_ratio=None, num_frames=num_frames),
        } 
    else:
        processor = {
                'image': partial(process_image, processor=processor, aspect_ratio=None),
                'video': partial(process_video, processor=processor, aspect_ratio=None, num_frames=num_frames),
                'face': partial(process_image_npary, processor=processor, aspect_ratio=None),
                'audio': partial(process_audio, processor=audio_processor),
            }
    return model, processor, tokenizer

def mm_infer_prepare_inputs(image_or_video, instruct, model, tokenizer, audio=None, modal='video', question=None, bert_tokeni=None, **kwargs):
    """Prepare all model inputs for HumanOmni inference (without running generate).

    Args:
        model: HumanOmni model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        audio (torch.Tensor, optional): optional audio input.
        modal (str): 'image', 'video', 'audio', 'video_audio', or 'text'.
        question (str, optional): optional question prompt.
        bert_tokeni: tokenizer for question (e.g., BERT tokenizer).
        **kwargs: additional parameters like do_sample, temperature, etc.
    
    Returns:
        dict: all inputs required by model.generate().
    """
    import torch, copy, transformers

    question_prompt = None
    if question is not None:
        question = [question]
        question_prompt = bert_tokeni(question, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
        question_prompt = {key: value.to('cuda') for key, value in question_prompt.items()}

    # --- determine modal token ---
    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    elif modal == 'video_audio':
        modal_token = DEFAULT_VIDEO_TOKEN + '\n' + DEFAULT_AUDIO_TOKEN
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")

    # --- vision preprocess ---
    if modal in ['text', 'audio']:
        if kwargs.get("model_size") == "7B":
            tensor = [(torch.zeros(32, 3, 384, 384).cuda().half(), "video")]
        else:
            tensor = [(torch.zeros(32, 3, 224, 224).cuda().half(), "video")]
    else:
        vi_modal = "video" if "video" in modal else "image"
        if isinstance(image_or_video, transformers.image_processing_base.BatchFeature):
            processed_data = transformers.image_processing_base.BatchFeature({
                'pixel_values_videos': image_or_video['pixel_values_videos'][0].half().cuda(),
                'video_grid_thw': image_or_video['video_grid_thw'][0].cuda()
            })
        else:
            processed_data = image_or_video.half().cuda()
        tensor = [(processed_data, vi_modal)]

    if audio is not None:
        audio = audio.half().cuda()

    # --- text preprocess ---
    if isinstance(instruct, str):
        
        content_template = [{'type': 'video',
          'video': kwargs['video_path']}, 
          {'text': modal_token + '\n' + instruct,
           'type': 'text'}]
        message = [{'role': 'user', 'content': content_template}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported instruct type: {type(instruct)}")

    # --- add system message ---
    if model.config.model_type in ['HumanOmni', 'HumanOmni_mistral', 'HumanOmni_mixtral']:
        system_message = [{
            'role': 'system',
            'content': (
                "<<SYS>>\nYou are a helpful, respectful and honest assistant..."
                "\n<</SYS>>"
            )
        }]
    else:
        system_message = []

    message = system_message + message

    # --- multimodal token wrapping ---
    if model.config.mm_use_x_start_end:
        message[0]['content'][1]['text'] = (
            message[0]['content'][1]['text'].replace("<video>", "<vi_start><video><vi_end>")
                .replace("<image>", "<im_start><image><im_end>")
                .replace("<audio>", "<au_start><audio><au_end>")
        )
    # if kwargs['model_name'] in ['ERV','R1']:
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    # else:
    #     prompt = message[0]['content'][1]['text']
    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    # --- return prepared inputs instead of generating ---
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "images": tensor,
        "prompts": question_prompt,
        "audios": audio,
        "generation_args": {
            "do_sample": kwargs.get('do_sample', False),
            "temperature": kwargs.get('temperature', 0.3 if kwargs.get('do_sample', False) else 0.0),
            "top_p": kwargs.get('top_p', 0.9),
            "max_new_tokens": kwargs.get('max_new_tokens', 2048),
            "stopping_criteria": [stopping_criteria],
            "pad_token_id": tokenizer.eos_token_id
        },
        "raw_prompt_text": prompt
    }

    return inputs

def mm_infer(image_or_video, instruct, model, tokenizer, audio=None, modal='video', question=None, bert_tokeni=None, **kwargs):
    """inference api of HumanOmni for video understanding.

    Args:
        model: HumanOmni model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """
    question_prompt = None
    if question is not None:
        question = [question]
        question_prompt = bert_tokeni(question, return_tensors='pt', padding=True, truncation=True,add_special_tokens=True)
        question_prompt = {key: value.to('cuda') for key, value in question_prompt.items()}

    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    elif modal == 'video_audio':
        modal_token = DEFAULT_VIDEO_TOKEN + '\n' +DEFAULT_AUDIO_TOKEN
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")


    # 1. vision preprocess (load & transform image or video).

    if modal == 'text' or modal == 'audio':
        if kwargs["model_size"] == "7B":
            tensor = [(torch.zeros(32, 3, 384, 384).cuda().half(), "video")]
        else: # 224 x 224
            tensor = [(torch.zeros(32, 3, 224, 224).cuda().half(), "video")]
    else:
        if "video" in modal:
            vi_modal = "video"
        else:
            vi_modal = "image"

        if isinstance(image_or_video, transformers.image_processing_base.BatchFeature):
            # 处理 BatchFeature 中的所有 tensor
            processed_data = transformers.image_processing_base.BatchFeature({
                'pixel_values_videos': image_or_video['pixel_values_videos'][0].half().cuda(),
                'video_grid_thw': image_or_video['video_grid_thw'][0].cuda()
            })
        else:
            # 处理普通 tensor
            processed_data = image_or_video.half().cuda()
        tensor = [(processed_data, vi_modal)]

    
    if audio is not None:
        audio = audio.half().cuda()

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")


    
    if model.config.model_type in ['HumanOmni', 'HumanOmni_mistral', 'HumanOmni_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    # add modal warpper tokken
    if model.config.mm_use_x_start_end:
        prompt = prompt.replace("<video>", "<vi_start><video><vi_end>").replace("<image>", "<im_start><image><im_end>").replace("<audio>", "<au_start><audio><au_end>")

    # prompt = prompt + '\n<think> </think>\n<answer>'
    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = kwargs.get('do_sample', False)
    if do_sample:
        temperature = kwargs.get('temperature', 0.3)
    else:
        temperature = 0.0

    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 512)


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            prompts=question_prompt,
            audios=audio
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs


def mm_infer_text_classifier(image_or_video, instruct, model, tokenizer, audio=None, modal='video', question=None, bert_tokeni=None, **kwargs):
    """inference api of HumanOmni for video understanding.

    Args:
        model: HumanOmni model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """
    question_prompt = None
    if question is not None:
        question = [question]
        question_prompt = bert_tokeni(question, return_tensors='pt', padding=True, truncation=True,add_special_tokens=True)
        question_prompt = {key: value.to('cuda') for key, value in question_prompt.items()}

    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    elif modal == 'video_audio':
        modal_token = DEFAULT_VIDEO_TOKEN + '\n' +DEFAULT_AUDIO_TOKEN
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")


    # 1. vision preprocess (load & transform image or video).

    if modal == 'text' or modal == 'audio':
        if kwargs["model_size"] == "7B":
            tensor = [(torch.zeros(32, 3, 384, 384).cuda().half(), "video")]
        else: # 224 x 224
            tensor = [(torch.zeros(32, 3, 224, 224).cuda().half(), "video")]
    else:
        if "video" in modal:
            vi_modal = "video"
        else:
            vi_modal = "image"

        if isinstance(image_or_video, transformers.image_processing_base.BatchFeature):
            # 处理 BatchFeature 中的所有 tensor
            processed_data = transformers.image_processing_base.BatchFeature({
                'pixel_values_videos': image_or_video['pixel_values_videos'][0].half().cuda(),
                'video_grid_thw': image_or_video['video_grid_thw'][0].cuda()
            })
        else:
            # 处理普通 tensor
            processed_data = image_or_video.half().cuda()
        tensor = [(processed_data, vi_modal)]

    
    if audio is not None:
        audio = audio.half().cuda()

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")


    
    if model.config.model_type in ['HumanOmni', 'HumanOmni_mistral', 'HumanOmni_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    # add modal warpper tokken
    if model.config.mm_use_x_start_end:
        prompt = prompt.replace("<video>", "<vi_start><video><vi_end>").replace("<image>", "<im_start><image><im_end>").replace("<audio>", "<au_start><audio><au_end>")

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            prompts=question_prompt,
            audios=audio
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
    reason_output = [outputs.split('<think>')[1].split('</think>')[0].strip() if '<think>' in outputs else '']

    em_inputs = EMOTION_TOKENIZER(reason_output,
                                  return_tensors="pt",
                                    padding=True,
                                    truncation=True).to(EMOTION_DEVICE)
    with torch.no_grad():
        logits = EMOTION_MODEL(**em_inputs).logits
    probs = torch.sigmoid(logits) 
    B, C = probs.shape
    
    threshold=0.5
    preds = (probs > threshold).int() 
    
    top_indices_list=[]
    for i in range(B):
        idxs = torch.topk(probs[i], k=2).indices.tolist()
        # idxs = (preds[i] == 1).nonzero(as_tuple=True)[0].tolist()
        # if len(idxs) == 0:
        #     idxs = [torch.argmax(probs[i]).item()]
        # elif len(idxs) > 2:
        #     idxs = torch.topk(probs[i], k=2).indices.tolist()

        top_indices_list.append(idxs)
    
    # for top_idx in top_indices_list[0]:
    #     if REVERSE_GOEMOTION_TO_LABEL[top_idx] in ['fear','disgust','anxiety','helplessness','disappointment','contempt']:
    #         answer= '<answer>' + REVERSE_GOEMOTION_TO_LABEL[top_idx] + '</answer>'
    #         output_answers = outputs.split('</think>')[-1].strip()
    #         outputs = outputs.replace(output_answers, answer)
    #         return outputs
    tmp_list = [] 
    for top_idx in top_indices_list[0]:
        try:
            tmp_list.append(REVERSE_GOEMOTION_TO_LABEL[top_idx])
        except:
            pass
    final_pred = '/'.join(tmp_list)
    return outputs, final_pred

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

def mm_batch_infer(image_or_video, instruct, model, tokenizer, audio=None, modal='video', question=None, bert_tokeni=None, **kwargs):
    """inference api of HumanOmni for video understanding.

    Args:
        model: HumanOmni model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """
    question_prompt = None
    if question is not None:
        question = [question]
        question_prompt = bert_tokeni(question, return_tensors='pt', padding=True, truncation=True,add_special_tokens=True)
        question_prompt = {key: value.to('cuda') for key, value in question_prompt.items()}

    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    elif modal == 'video_audio':
        modal_token = DEFAULT_VIDEO_TOKEN + '\n' +DEFAULT_AUDIO_TOKEN
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")


    # 1. vision preprocess (load & transform image or video).

    if modal == 'text' or modal == 'audio':
        if kwargs["model_size"] == "7B":
            tensor = [(torch.zeros(32, 3, 384, 384).cuda().half(), "video")]
        else:
            tensor = [(torch.zeros(32, 3, 224, 224).cuda().half(), "video")]
    else:
        if "video" in modal:
            vi_modal = "video"
        else:
            vi_modal = "image"

        if isinstance(image_or_video, transformers.image_processing_base.BatchFeature):
            # 处理 BatchFeature 中的所有 tensor
            processed_data = transformers.image_processing_base.BatchFeature({
                'pixel_values_videos': image_or_video['pixel_values_videos'][0].half().cuda(),
                'video_grid_thw': image_or_video['video_grid_thw'][0].cuda()
            })
        else:
            # 处理普通 tensor
            processed_data = image_or_video.half().cuda()
        tensor = [(processed_data, vi_modal)]

    
    if audio is not None:
        audio = audio.half().cuda()

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")


    
    if model.config.model_type in ['HumanOmni', 'HumanOmni_mistral', 'HumanOmni_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    # add modal warpper tokken
    if model.config.mm_use_x_start_end:
        prompt = prompt.replace("<video>", "<vi_start><video><vi_end>").replace("<image>", "<im_start><image><im_end>").replace("<audio>", "<au_start><audio><au_end>")

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            prompts=question_prompt,
            audios=audio
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True) #[0].strip()

    return outputs


def prepare_mm_inputs(image_or_video, instruct, model, tokenizer, audio=None, modal='video', question=None, bert_tokeni=None, **kwargs):
    """inference api of HumanOmni for video understanding.

    Args:
        model: HumanOmni model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """
    question_prompt = None
    if question is not None:
        question = [question]
        question_prompt = bert_tokeni(question, return_tensors='pt', padding=True, truncation=True,add_special_tokens=True)
        question_prompt = {key: value.to('cuda') for key, value in question_prompt.items()}

    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    elif modal == 'video_audio':
        modal_token = DEFAULT_VIDEO_TOKEN + '\n' +DEFAULT_AUDIO_TOKEN
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")


    # 1. vision preprocess (load & transform image or video).

    if modal == 'text' or modal == 'audio':
        tensor = [(torch.zeros(32, 3, 384, 384).cuda().half(), "video")]
    else:
        if "video" in modal:
            vi_modal = "video"
        else:
            vi_modal = "image"

        if isinstance(image_or_video, transformers.image_processing_base.BatchFeature):
            processed_data = transformers.image_processing_base.BatchFeature({
                'pixel_values_videos': image_or_video['pixel_values_videos'][0].half().cuda(),
                'video_grid_thw': image_or_video['video_grid_thw'][0].cuda()
            })
        else:
            processed_data = image_or_video.half().cuda()
        tensor = [(processed_data, vi_modal)]

    
    if audio is not None:
        audio = audio.half().cuda()

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")


    
    if model.config.model_type in ['HumanOmni', 'HumanOmni_mistral', 'HumanOmni_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    # add modal warpper tokken
    if model.config.mm_use_x_start_end:
        prompt = prompt.replace("<video>", "<vi_start><video><vi_end>").replace("<image>", "<im_start><image><im_end>").replace("<audio>", "<au_start><audio><au_end>")

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    return input_ids, tensor, audio, question_prompt



def mm_infer_add_description(image_or_video, instruct, model, tokenizer, audio=None, modal='video', question=None, bert_tokeni=None, description=None, use_description=True, **kwargs):
    """inference api of HumanOmni for video understanding.

    Args:
        model: HumanOmni model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """
    question_prompt = None
    if question is not None:
        question = [question]
        question_prompt = bert_tokeni(question, return_tensors='pt', padding=True, truncation=True,add_special_tokens=True)
        question_prompt = {key: value.to('cuda') for key, value in question_prompt.items()}

    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    elif modal == 'video_audio':
        modal_token = DEFAULT_VIDEO_TOKEN + '\n' +DEFAULT_AUDIO_TOKEN
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")


    # 1. vision preprocess (load & transform image or video).

    if modal == 'text' or modal == 'audio':
        if kwargs["model_size"] == "7B":
            tensor = [(torch.zeros(32, 3, 384, 384).cuda().half(), "video")]
        else: # 224 x 224
            tensor = [(torch.zeros(32, 3, 224, 224).cuda().half(), "video")]
    else:
        if "video" in modal:
            vi_modal = "video"
        else:
            vi_modal = "image"

        if isinstance(image_or_video, transformers.image_processing_base.BatchFeature):
            # 处理 BatchFeature 中的所有 tensor
            processed_data = transformers.image_processing_base.BatchFeature({
                'pixel_values_videos': image_or_video['pixel_values_videos'][0].half().cuda(),
                'video_grid_thw': image_or_video['video_grid_thw'][0].cuda()
            })
        else:
            # 处理普通 tensor
            processed_data = image_or_video.half().cuda()
        tensor = [(processed_data, vi_modal)]

    
    if audio is not None:
        audio = audio.half().cuda()

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")


    
    if model.config.model_type in ['HumanOmni', 'HumanOmni_mistral', 'HumanOmni_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    # add modal warpper tokken
    if model.config.mm_use_x_start_end:
        prompt = prompt.replace("<video>", "<vi_start><video><vi_end>").replace("<image>", "<im_start><image><im_end>").replace("<audio>", "<au_start><audio><au_end>")

    # Add description to the prompt
    if use_description:
        prompt = prompt + '\n<think>' + description + '</think>\n<answer>'

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            prompts=question_prompt,
            audios=audio
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    
    # 검증 과정
    reason_output = [outputs.split('<think>')[1].split('</think>')[0].strip() if '<think>' in outputs else '']
    em_inputs = EMOTION_TOKENIZER(reason_output,
                                  return_tensors="pt",
                                    padding=True,
                                    truncation=True).to(EMOTION_DEVICE)
    with torch.no_grad():
        logits = EMOTION_MODEL(**em_inputs).logits
    probs = F.softmax(logits, dim=1).squeeze()
    topk = torch.topk(probs, k=1)
    top_indices = topk.indices.tolist()

    tmp_list = []
    for index in top_indices:
        tmp_list.append(match_emotion_EMOBERT(EMOTION_MODEL.config.id2label[index]))
    
    if topk.values.item() <0.9:
        prompt = prompt + '\n<think> </think>\n<answer>'
        input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()
        keywords = [tokenizer.eos_token]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_masks,
                images=tensor,
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=tokenizer.eos_token_id,
                prompts=question_prompt,
                audios=audio
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    else:
        return outputs
    


def mm_infer_conf(image_or_video, instruct, model, tokenizer, audio=None, modal='video', question=None, bert_tokeni=None, **kwargs):
    """inference api of HumanOmni for video understanding.

    Args:
        model: HumanOmni model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """
    def _find_subseq(haystack, needle):
        """haystack(list[int])에서 needle(list[int]) 첫 등장 시작 인덱스 반환, 없으면 -1"""
        if len(needle) == 0 or len(haystack) < len(needle):
            return -1
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i:i+len(needle)] == needle:
                return i
        return -1
    
    
    question_prompt = None
    if question is not None:
        question = [question]
        question_prompt = bert_tokeni(question, return_tensors='pt', padding=True, truncation=True,add_special_tokens=True)
        question_prompt = {key: value.to('cuda') for key, value in question_prompt.items()}

    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    elif modal == 'video_audio':
        modal_token = DEFAULT_VIDEO_TOKEN + '\n' +DEFAULT_AUDIO_TOKEN
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")


    # 1. vision preprocess (load & transform image or video).

    if modal == 'text' or modal == 'audio':
        if kwargs["model_size"] == "7B":
            tensor = [(torch.zeros(32, 3, 384, 384).cuda().half(), "video")]
        else: # 224 x 224
            tensor = [(torch.zeros(32, 3, 224, 224).cuda().half(), "video")]
    else:
        if "video" in modal:
            vi_modal = "video"
        else:
            vi_modal = "image"

        if isinstance(image_or_video, transformers.image_processing_base.BatchFeature):
            # 处理 BatchFeature 中的所有 tensor
            processed_data = transformers.image_processing_base.BatchFeature({
                'pixel_values_videos': image_or_video['pixel_values_videos'][0].half().cuda(),
                'video_grid_thw': image_or_video['video_grid_thw'][0].cuda()
            })
        else:
            # 处理普通 tensor
            processed_data = image_or_video.half().cuda()
        tensor = [(processed_data, vi_modal)]

    
    if audio is not None:
        audio = audio.half().cuda()

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")


    
    if model.config.model_type in ['HumanOmni', 'HumanOmni_mistral', 'HumanOmni_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    # add modal warpper tokken
    if model.config.mm_use_x_start_end:
        prompt = prompt.replace("<video>", "<vi_start><video><vi_end>").replace("<image>", "<im_start><image><im_end>").replace("<audio>", "<au_start><audio><au_end>")


    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    labels_tuple = ("angry","anger","sad","sadness","happy","happiness","neutral","surprise","surprised","disgust","disgusted","fear","fearful","anxiety","anxious","helplessness","helpless","disappointment","disappointed","contempt","contemptuous")

    with torch.inference_mode():
        gen_out = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            prompts=question_prompt,
            audios=audio,
            return_dict_in_generate=True,
            output_scores=True,          # 각 스텝의 vocab 로짓
        )
    # 1. output ids -> text
    sequences = gen_out.sequences # Response부터의 출력
    outputs = tokenizer.decode(sequences[0], skip_special_tokens=True)

    print("outputs:", outputs)
    print("sequences:", sequences.shape)

    if gen_out.scores is not None and len(gen_out.scores) > 0:
        gen_logits = torch.cat(gen_out.scores,dim=0) # (gen_len, vocab)
        gen_logits_cpu = gen_logits.detach().cpu()    # (gen_len, vocab)
    else:
        gen_logits_cpu = None
    
    # 3. 특정 Token logits 확인
    gen_ids_list = sequences[0].tolist()
    start_tag_ids = tokenizer.encode("<answer>", add_special_tokens=False)
    end_tag_ids   = tokenizer.encode("</answer>", add_special_tokens=False)
    
    start_idx = _find_subseq(gen_ids_list, start_tag_ids)
    end_idx = _find_subseq(gen_ids_list, end_tag_ids) if start_idx != -1 else -1
    answer_found = (start_idx != -1 and end_idx != -1 and end_idx >= start_idx + len(start_tag_ids))
    
    if answer_found and gen_logits is not None:
        # 태그 내부 구간: "<answer>" 직후부터 "</answer>" 시작 직전까지
        inner_start = start_idx + len(start_tag_ids)
        inner_end_exclusive = end_idx

        answer_token_ids = sequences[0, inner_start:inner_end_exclusive].detach().cpu()
        answer_logits = gen_logits[inner_start:inner_end_exclusive, :]  # (Len-Answer, vocab)
        answer_softmax = F.softmax(answer_logits, dim=-1)  # (Len-Answer, vocab)

        # 선택된 토큰 로짓만(스칼라 벡터)도 함께 제공
        answer_chosen_softmax = answer_softmax[:, answer_token_ids]
        best_answer = tokenizer.decode(answer_token_ids)
        print('best_answer:', best_answer)
        print('answer chosen_softmax:', answer_chosen_softmax)
        confidence = answer_chosen_softmax[0].max().item()
        print('confidence:', confidence)
        answer_span_text = tokenizer.decode(answer_token_ids, skip_special_tokens=True)
        answer_span_indices = (inner_start, inner_end_exclusive)
    else:
        # 태그가 없거나 로짓 없음
        answer_token_ids = torch.tensor([], dtype=torch.long)
        answer_logits = torch.empty((0, getattr(model.config, "vocab_size", 0)))
        answer_chosen_token_logits = torch.tensor([], dtype=torch.float32)
        answer_span_text = ""
        answer_span_indices = (-1, -1)
        confidence = 0

    print('answer_span_text:', answer_span_text)

    return outputs, confidence
    