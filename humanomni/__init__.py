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



def model_init(model_path=None, **kwargs):
    # with_face = kwargs.get('with_face', False)
    model_path = "HumanOmni_7B" if model_path is None else model_path
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, processor, context_len, audio_processor = load_pretrained_model(model_path, None, model_name, **kwargs)

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES
    processor = {
            'image': partial(process_image, processor=processor, aspect_ratio=None),
            'video': partial(process_video, processor=processor, aspect_ratio=None, num_frames=num_frames),
            'face': partial(process_image_npary, processor=processor, aspect_ratio=None),
            'audio': partial(process_audio, processor=audio_processor),
        }
    return model, processor, tokenizer

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
    