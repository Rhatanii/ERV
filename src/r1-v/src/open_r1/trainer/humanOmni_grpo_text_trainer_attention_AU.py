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

import io
from contextlib import redirect_stdout


import os
import re
import json
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from qwen_vl_utils import process_vision_info
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from transformers.trainer_utils import EvalPrediction
import numpy as np

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

import copy
from transformers import BertModel, BertTokenizer

import sys
sys.path.append("/mnt/ssd_hs/Exp/R1-Omni")
from humanomni.model import *
from humanomni.constants import NUM_FRAMES, IGNORE_INDEX, MODAL_INDEX_MAP, DEFAULT_X_START_TOKEN, DEFAULT_X_END_TOKEN
from humanomni.mm_utils import tokenizer_multimodal_token, process_image, process_video, read_video_patch, process_audio, frame_sample,get_model_name_from_path
from humanomni import model_init, mm_infer
from humanomni.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN
from transformers import (
    CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig,
    SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig,
     WhisperFeatureExtractor, WhisperProcessor, WhisperConfig, WhisperForAudioClassification
)
import os
sys.path.append('/mnt/ssd_hs/Exp/R1-Omni/src/r1-v')
from src.open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer
# sys.path.append('/mnt/data/jiaxing.zjx/code/HumanOmni/')
# sys.path.append('/mnt/data/jiaxing.zjx/cache/huggingface/')
#初始化BERT分词器
# bert_model = "bert-base-uncased"
# bert_tokenizer = BertTokenizer.from_pretrained(bert_model)

from contextlib import suppress

def check_parameters(model, log_file: str | None = None):
    """
    Works both before and after DeepSpeed ZeRO‑3 wrapping.
    Saves the same console output to `log_file` when given.
    """
    from deepspeed.runtime.zero.partition_parameters import GatheredParameters  # lazily import

    def _print(*args, **kwargs):
        print(*args, **kwargs)
        if log_file is not None:
            print(*args, **kwargs, file=fp)

    frozen, trainable = [], []
    total_numel, trainable_numel = 0, 0

    # open log file if requested
    with (open(log_file, "w") if log_file else suppress()) as fp:
        for name, param in model.named_parameters():
            # ── 파라미터 전체 크기 얻기 ─────────────────────
            if hasattr(param, "ds_numel"):          # ZeRO‑3 Parameter
                numel = param.ds_numel
            else:                                   # 일반 Parameter
                numel = param.numel()

            # 일부 rank는 shard 크기가 0일 수 있으므로, 필요하면 모아서 측정
            if numel == 0:
                with GatheredParameters([param]):
                    numel = param.numel()

            total_numel += numel
            if param.requires_grad:
                trainable.append((name, numel))
                trainable_numel += numel
            else:
                frozen.append((name, numel))

        pct = 100 * trainable_numel / total_numel if total_numel else 0

        # ── 출력 ─────────────────────────────────────────
        _print("Frozen Parameters:")
        for n, _ in frozen:
            _print(f"  {n}")

        _print("\nTrainable Parameters:")
        for n, _ in trainable:
            _print(f"  {n}")

        _print("\n────────────  Summary  ────────────")
        _print(f"Total parameters     : {total_numel:,d}")
        _print(f"Trainable parameters : {trainable_numel:,d}  ({pct:.4f}%)")
        _print(f"Frozen parameters    : {total_numel - trainable_numel:,d}")

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

from moviepy.editor import VideoFileClip

def get_video_duration(video_path):
    """Get the duration of a video file."""
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception as e:
        print(f"Error reading video file {video_path}: {e}")
        return None

class SkipSampler(Sampler):
    def __init__(self, base_sampler, skip_count=0):
        self.base_sampler = base_sampler
        self.skip_count = skip_count

    def __iter__(self):
        iterator = iter(self.base_sampler)
        # skip the first N items
        for _ in range(self.skip_count):
            try:
                next(iterator)
            except StopIteration:
                break
        for idx in iterator:
            yield idx

    def __len__(self):
        return len(self.base_sampler) - self.skip_count

class HumanOmniVLGRPO_TEXT_Attention_AU_Trainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        select_layer_idx: Optional[Union[int, str]] = 18,
        reference_model_switch: Optional[bool] = True,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
       # import ipdb;ipdb.set_trace()
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        
        if isinstance(model, str):
            model_id = model
            model_name_d = model_id
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            
            # Quantization
            if quantization_config is not None:
                # default to 4-bit NF4
                bnb_config = quantization_config
            else:
                bnb_config = None
                
                
            config = VLLMConfigs["HumanOmni_qwen2"].from_pretrained(
                model, 
                trust_remote_code=True,
                use_fast = True
            )
            config.mm_vision_tower = '/mnt/ssd_hs/Exp/R1-Omni/pre-trained/siglip-base-patch16-224'
            config.mm_audio_tower = '/mnt/ssd_hs/Exp/R1-Omni/pre-trained/whisper-large-v3'
            model = VLLMs["HumanOmni_qwen2"].from_pretrained(
                model,
                config=config,
                quantization_config=bnb_config,
                cache_dir=None,
                torch_dtype=torch.bfloat16,
                do_sample=True,
                use_fast = True
            )
            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()

            audio_tower = model.get_audio_tower()
            if not audio_tower.is_loaded:
                audio_tower.load_model()

            audio_tower = model.get_audio_tower()
            self.audio_processor = WhisperFeatureExtractor.from_pretrained(config.mm_audio_tower, use_fast=True)

            vision_tower = model.get_vision_tower()
            self.visual_processor = SiglipImageProcessor.from_pretrained(config.mm_vision_tower, use_fast = True)


        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
                
        # with open('/mnt/ssd_hs/Exp/R1-Omni/output_0.5B.txt','w') as wf:
        #     for n,p in model.named_parameters():
        #         if p.requires_grad:
        #             print(f"Trainable parameter: {n}", file=wf)
        #         else:
        #             print(f"Frozen parameter: {n}", file=wf)
        # peft
        # if peft_config is not None:
        #     model = get_peft_model(model, peft_config)
        #     model.print_trainable_parameters()
        # else:
        #     # If the model is not wrapped with PEFT, we can still check the parameters
        #     check_parameters(model)
            

        if peft_config is not None:
            model = get_peft_model(model, peft_config)
            check_parameters(model, log_file="/mnt/ssd_hs/Exp/R1-Omni/trainable_params_peft.txt")
        else:
            check_parameters(model, log_file="/mnt/ssd_hs/Exp/R1-Omni/trainable_params.txt")
        
        # Reference model
        self.reference_model_switch = reference_model_switch
        if self.reference_model_switch:
            self.ref_model = VLLMs["HumanOmni_qwen2"].from_pretrained(
                model_name_d,
                config=config,
                cache_dir=None,
                torch_dtype=torch.bfloat16,
                do_sample=True,
                use_fast = True
            )
            vision_tower = self.ref_model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()

            audio_tower = self.ref_model.get_audio_tower()
            if not audio_tower.is_loaded:
                audio_tower.load_model()
        else:
            self.ref_model = None

        bert_model = "/mnt/ssd_hs/Exp/R1-Omni/pre-trained/bert-base-uncased"
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model, use_fast = True)


        # Processing class
        if processing_class is None:
            processing_class = processing_class = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", use_fast = True)
            pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.pad_token_id = pad_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path, use_fast=True)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1, 
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta
        self.select_layer_idx = select_layer_idx
        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)


    def _build_generation_inputs(self, inputs):
            """
            inputs: list[dict]  (데이터로더가 넘겨주는 원시 예시)
            return:
            prompt_inputs ........  dict( inputs / images / audios / ... )
            prompt_ids, prompt_mask
            """
            prompts = []
            bert_prompts = []
            for x in inputs:
                prompt = x["prompt"]
                bert_prompts.append(prompt[0]['content'][1]['text'])
                text = prompt[0]["content"][0].pop('text')
                video_path = x["video"]
                prompt[0]["content"][0]["video"] = x["video"]
                prompt[0]['content'][1]['text'] = ('<vi_start><video><vi_end>\n'
                                                '<au_start><audio><au_end>\n'
                                                + prompt[0]['content'][1]['text'])
                prompts.append(prompt)

            # ------------- BERT sub-encoder -------------
            bert_prompts = self.bert_tokenizer(
                bert_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                add_special_tokens=True,
            )

            # ------------- LLM 텍스트 -------------
            prompts_text = [
                maybe_apply_chat_template(ex, self.processing_class)["prompt"]
                for ex in inputs
            ]
            input_ids = [
                tokenizer_multimodal_token(p_txt, self.processing_class.tokenizer,
                                        '<video>', return_tensors='pt')
                for p_txt in prompts_text
            ]
            input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
            attention_masks = input_ids.ne(self.processing_class.pad_token_id)

            # ------------- Video / Audio -------------
            videos, audios = [], []
            for prompt in prompts:
                video_file = prompt[0]["content"][0]["video"]
                videos.append(
                    process_video(video_file, self.visual_processor,
                                aspect_ratio="pad", num_frames=8)
                )
                audio, sr = process_audio(video_file)
                audio_feats = self.audio_processor(
                    audio, sampling_rate=sr,
                    return_tensors='pt')['input_features']
                audios.append(audio_feats)

            videos = torch.cat(videos, dim=0).unsqueeze(0)
            audios = torch.cat(audios, dim=0) #.unsqueeze(0)

            # ------------- pack & move to device -------------
            prompt_inputs = {
                'inputs':          input_ids,
                'attention_mask':  attention_masks,
                'images':          videos,
                'audios':          audios,
                'prompts':         bert_prompts,
            }
            prompt_inputs = super()._prepare_inputs(prompt_inputs)

            # 일부 downstream 코드가 id/mask 길이를 참조하므로 리턴
            return prompt_inputs, prompt_inputs["inputs"], prompt_inputs["attention_mask"]

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    def _get_per_token_logps_video(
    self, model, input_ids, attention_mask, images, audios, prompts, answer_length, layer_idx: Optional[Union[int, str]] = None, capture_attention: bool = True,
    prompt_length: int = None,         # compute_loss에서 넘겨줌
    attn_dtype=torch.float16,
    offload_to_cpu: bool = False,
    ):
        """
            Returns:
        per_token_logps: (B, answer_length)
        attn_pack: dict or None
            - "last_mean": (B, answer_length, L)  # head-mean, 마지막 레이어
            - "to_prompt_sum": (B, answer_length) # 프롬프트 구간으로의 attention 합
            - "to_prompt_mean": (B, answer_length)
            - "entropy": (B, answer_length)       # 각 위치의 attention 분포 엔트로피
        """
        attn_pack = None

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            images=images,
            audios=audios,
            prompts=prompts,
            output_attentions=capture_attention,  # 여기서만 True
            use_cache=False,
            return_dict=True,
        )

        logits = outputs.logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        
        logits = logits[:, (-answer_length) :]
        input_ids = input_ids[:, (-answer_length) :]
        
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        per_token_logps = torch.stack(per_token_logps)
        
        # --- attention 추출/정리 ---
        if capture_attention:
            # outputs.attentions: tuple(n_layers) of (B, H, L, L)
            if layer_idx == 'mean':
                last = torch.stack(outputs.attentions, dim=0).mean(dim=0)  # (n_layers, B, H, L, L) -> (B, H, L, L)
            else:
                last = outputs.attentions[layer_idx]  # (B, H, L, L)
            # last = last[:, :, :-1, :]  # 마지막 포지션 제외

            # 정답 구간 쿼리만 남기기
            last_ans = last[:, :, (-answer_length):, :]  # (B, H, answer, L)
            # head 평균 + dtype 축소
            last_mean = last_ans.mean(dim=1).to(attn_dtype)  # (B, answer, L)


            # 프롬프트로의 attention 통계
            vis_length = 392
            vis_left_length = 7
            vis_right_length = 4
            
            aud_length = 500
            aud_left_length = 4
            aud_right_length = 4
            
            # prompt_length = 51
            
            vis_start_idx = 14
            aud_start_idx = vis_start_idx + vis_left_length + vis_length + vis_right_length
            prompt_start_idx = aud_start_idx + aud_left_length + aud_length + aud_right_length

            # reason_start_idx = 976
            reason_start_idx = prompt_length -2 + vis_length + aud_length  
            
            # reason_divisor = torch.arange(reason_start_idx, reason_start_idx + answer_length, dtype=last_mean.dtype, device=last_mean.device)
            # reason_numerator = torch.arange(1, 1 + answer_length, dtype=last_mean.dtype, device=last_mean.device)
            prompt_indices = torch.cat((
                torch.arange(0, vis_start_idx),
                torch.arange(prompt_start_idx, reason_start_idx)
            ))
            to_prompt = last_mean[:, :, prompt_indices].sum(dim=-1)  # (B, answer), P
            to_vision = last_mean[:, :, vis_start_idx:aud_start_idx].sum(dim=-1)  # (B, answer), vis
            to_audio = last_mean[:, :, aud_start_idx:prompt_start_idx].sum(dim=-1)  # (B, answer), aud
            to_reason = last_mean[:, :, reason_start_idx:].sum(dim=-1)
            

            attn_pack = {
                "last_mean": last_mean,             # (B, answer, L)
                "to_prompt_mean": to_prompt,     # (B, answer)
                "to_vision_mean": to_vision,     # (B, answer)
                "to_audio_mean": to_audio,         # (B, answer)
                "to_reason_mean": to_reason,     # (B, answer)
            }
            # 큰 텐서는 즉시 참조 해제해 메모리 압력 완화
            del last, last_ans, last_mean, to_prompt, to_vision, to_audio, to_reason, outputs, # to_prompt_norm, to_vision_norm, to_audio_norm, to_reason_norm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return per_token_logps if attn_pack is None else (per_token_logps, attn_pack)


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        def pad_token_tensors(tensor_list, pad_token_id=self.processing_class.tokenizer.pad_token_id, padding_side='left'):
            """
            shape (L_i,) 텐서들이 담긴 리스트를 받아,
            가장 긴 길이에 맞춰 pad_token_id로 패딩한 (n, max_len) 텐서를 반환

            Args:
                tensor_list (list[torch.Tensor]): shape (L_i,) 텐서 리스트
                pad_token_id (int): 패딩에 사용할 토큰 ID (예: tokenizer.pad_token_id)

            Returns:
                torch.Tensor: shape (n, max_len)
            """
            # 각 텐서의 길이 계산
            lengths = [t.size(0) for t in tensor_list]
            max_len = max(lengths)

            padded_tensors = []
            for t in tensor_list:
                pad_len = max_len - t.size(0)
                if pad_len > 0:
                    if padding_side == 'left':
                        t_padded = F.pad(t, (pad_len, 0), value=pad_token_id)
                    else:  # 'right'
                        t_padded = F.pad(t, (0, pad_len), value=pad_token_id)
                else:
                    t_padded = t
                padded_tensors.append(t_padded)

            # (n, max_len)으로 스택
            return torch.stack(padded_tensors)
        
        
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompts = []
        bert_prompts = []
        prompts_text = []
        
        for x in inputs:
            prompt = x["prompt"]
            bert_prompts.append(prompt[0]['content'][1]['text'])
            # text = prompt[0]["content"][0].pop('text')
            # video_path = x["video"]
            prompt[0]["content"][0]["video"] = x["video"]
            prompt[0]['content'][1]['text'] = '<vi_start><video><vi_end>\n<au_start><audio><au_end>\n' + prompt[0]['content'][1]['text']
            prompts.append(prompt)
            
            # LLM Chat format prompt text
            prompt_text = maybe_apply_chat_template(x, self.processing_class)["prompt"]
            prompts_text.append(prompt_text)
        
        
        # Pass Instruction to BERT for projector regularizer
        bert_prompts = self.bert_tokenizer(bert_prompts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
        input_ids = [tokenizer_multimodal_token(prompts_text_, self.processing_class.tokenizer, '<video>', return_tensors='pt') for prompts_text_ in prompts_text]
        # input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
        # input_ids = torch.stack(input_ids, dim=0) # edit
        input_ids = pad_token_tensors(input_ids, pad_token_id=self.processing_class.tokenizer.pad_token_id)
        
        video = []
        audios = []
        for prompt in prompts:
            video_file = prompt[0]["content"][0]["video"]
            video_ids = process_video(video_file, self.visual_processor, aspect_ratio="pad", num_frames=8)
            video.append(video_ids)

            audio, audio_sample_rate = process_audio(video_file)
            audio = self.audio_processor(audio, sampling_rate=audio_sample_rate, return_tensors='pt')['input_features']
            audios.append(audio)
        # video = torch.cat(video, dim=0).unsqueeze(0)
        # audios = torch.cat(audios, dim=0).unsqueeze(0)
        video = torch.stack(video, dim=0) # edit (bs,8,3,224,224)
        audios = torch.cat(audios, dim=0) # edit

        attention_masks = input_ids.ne(self.processing_class.pad_token_id)
        prompt_inputs = {}
        prompt_inputs['inputs'] = input_ids
        prompt_inputs['images'] = video 
        prompt_inputs['attention_mask'] = attention_masks
        prompt_inputs['prompts'] = bert_prompts
        prompt_inputs['audios'] = audios

        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["inputs"], prompt_inputs["attention_mask"]
  
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            prompt_length = prompt_ids.size(1)
            answer_length = prompt_completion_ids.size(1)
            completion_ids = prompt_completion_ids
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            prompt_ids_repeat = prompt_ids.repeat_interleave(self.num_generations, dim=0)
            
        if self.state.global_step % 10 == 0:
            self.accelerator.print(f"[DEBUG] Batch size: {len(prompts)}")
            self.accelerator.print(f"[DEBUG] Example prompt text: {prompts_text[0]}")
            for sample_id_name in prompts:
                self.accelerator.print(f"[DEBUG] Example video path: {sample_id_name[0]['content'][0]['video']}")
            
            self.accelerator.print(f"[DEBUG] Step {self.state.global_step}")
            self.accelerator.print(f"[DEBUG] prompt_completion_ids shape: {prompt_completion_ids.shape}")
            
            lengths = [int((ids != self.processing_class.tokenizer.pad_token_id).sum()) for ids in prompt_completion_ids]
            max_idx = int(torch.argmax(torch.tensor(lengths)))
            min_idx = int(torch.argmin(torch.tensor(lengths)))
            
            shortest_text = self.processing_class.tokenizer.decode(
                prompt_completion_ids[min_idx], skip_special_tokens=True
            )
            self.accelerator.print(f"[DEBUG] Shortest output (len={lengths[min_idx]}): {shortest_text}")

            # 🔹 가장 긴 출력
            longest_text = self.processing_class.tokenizer.decode(
                prompt_completion_ids[max_idx], skip_special_tokens=True
            )
            self.accelerator.print(f"[DEBUG] Longest output (len={lengths[max_idx]}): {longest_text}")


        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()


        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        
        prompt_completion_ids_repeat = torch.cat([prompt_ids_repeat, prompt_completion_ids], dim=1)

        images_repeat = prompt_inputs['images'].repeat_interleave(self.num_generations, dim=0)
        audios_repeat = prompt_inputs['audios'].repeat_interleave(self.num_generations, dim=0)
        prompts_repeat = {}
        prompts_repeat['input_ids'] =  prompt_inputs['prompts']['input_ids'].repeat_interleave(self.num_generations, dim=0)
        prompts_repeat['token_type_ids'] =  prompt_inputs['prompts']['token_type_ids'].repeat_interleave(self.num_generations, dim=0)
        prompts_repeat['attention_mask'] =  prompt_inputs['prompts']['attention_mask'].repeat_interleave(self.num_generations, dim=0)
      
        check_capture_attention = False
        for reward_func in self.reward_funcs:
            if 'attention' in reward_func.__name__:
                check_capture_attention = True
                break
        layer_idx = self.select_layer_idx
        out= self._get_per_token_logps_video(model, prompt_completion_ids_repeat, attention_mask, images_repeat, audios_repeat, prompts_repeat, answer_length, layer_idx, check_capture_attention, prompt_length=prompt_length)
        
        if isinstance(out, tuple):
            per_token_logps, attn_pack = out
        else:
            per_token_logps, attn_pack = out, None
  
        if self.ref_model is not None:
            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps_video(self.ref_model, prompt_completion_ids_repeat, attention_mask, images_repeat, audios_repeat, prompts_repeat, answer_length, layer_idx, capture_attention=False, prompt_length=prompt_length)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        if use_image:
                            ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
                        if use_video:
                            ref_per_token_logps = self._get_per_token_logps_video(model, prompt_completion_ids, prompt_length, attention_mask, pixel_values_videos, video_grid_thw)
            ref_per_token_logps = ref_per_token_logps

            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                        
                # --- attention 점수 전달: 평균 값과 시퀀스별 값 모두 옵션으로 제공 ---
                if attn_pack is not None:
                    # (B*G, answer)로 맞추기
                    attn_to_prompt_mean_seq = attn_pack["to_prompt_mean"]  # (B*G, answer_len)
                    attn_to_vision_mean_seq = attn_pack["to_vision_mean"]
                    attn_to_audio_mean_seq = attn_pack["to_audio_mean"]
                    attn_to_reason_mean_seq = attn_pack["to_reason_mean"]
                    # 시퀀스 단위 스칼라(토큰 평균)
                    # attn_to_prompt_norm = attn_pack["to_prompt_norm"]
                    # attn_to_vision_norm = attn_pack["to_vision_norm"]
                    # attn_to_audio_norm = attn_pack["to_audio_norm"]
                    # attn_to_reason_norm = attn_pack["to_reason_norm"]

                    reward_kwargs["attention_prompt_scores"] = attn_to_prompt_mean_seq.detach().float().cpu()
                    reward_kwargs["attention_vision_scores"] = attn_to_vision_mean_seq.detach().float().cpu()
                    reward_kwargs["attention_audio_scores"] = attn_to_audio_mean_seq.detach().float().cpu()
                    reward_kwargs["attention_reason_scores"] = attn_to_reason_mean_seq.detach().float().cpu()
                    # reward_kwargs["attention_prompt_norm"] = attn_to_prompt_norm.detach().float().cpu()
                    # reward_kwargs["attention_vision_norm"] = attn_to_vision_norm.detach().float().cpu()
                    # reward_kwargs["attention_audio_norm"] = attn_to_audio_norm.detach().float().cpu()
                    # reward_kwargs["attention_reason_norm"] = attn_to_reason_norm.detach().float().cpu()

                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        if self.ref_model is not None:
            per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        else:
            per_token_loss = -per_token_loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        if self.ref_model is not None:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        if torch.cuda.is_available():
            from accelerate import Accelerator
            accelerator = getattr(self, "accelerator", None)
            if accelerator is not None:
                accelerator.wait_for_everyone()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return loss


    def compute_metrics(self, eval_pred: EvalPrediction):
            """
            eval_pred.predictions : np.ndarray (prediction_step()이 돌려준 logits 또는 토큰 IDs)
            eval_pred.label_ids   : np.ndarray (prediction_step()이 돌려준 labels)
            """
            preds, labels = eval_pred

            # ── 예시 1) completion 길이
            completion_len = (preds != self.processing_class.eos_token_id).sum(1).mean()

            # ── 예시 2) reward 함수 재사용 ─────────────────────
            decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=True) if labels is not None else None

            # self.reward_funcs 는 compute_loss 에서 이미 쓰던 리스트
            rewards = []
            for rf in self.reward_funcs:
                if callable(rf):
                    r = rf(prompts=[None]*len(decoded_preds),   # 필요 매개변수 맞춰 전달
                        completions=[[{"content": p}] for p in decoded_preds])
                else:                                         # PreTrainedModel 타입
                    # … 로짓 모델일 때 처리 …
                    r = [0.0]*len(decoded_preds)
                rewards.append(np.mean(r))

            metrics = {
                "eval_completion_length": completion_len.item(),
            }
            for name, val in zip([getattr(rf, "__name__", "model") for rf in self.reward_funcs], rewards):
                metrics[f"eval_reward/{name}"] = val

            return metrics
    
    def prediction_step(
        self,
        model,
        inputs,                     # list[dict]
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        # 1) 전처리 재사용
        prompt_inputs, prompt_ids, prompt_mask = self._build_generation_inputs(inputs)

        # 2) truncate 등 동일
        if self.max_prompt_length is not None:
            prompt_ids  = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # 3) 모델 꺼내기 (hooks 필요 없음)
        unwrapped_model = self.accelerator.unwrap_model(model)

        # 4) generate
        with torch.no_grad():
            gen_ids = unwrapped_model.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
            )
        logits = gen_ids.detach()
        logits = logits.contiguous() 
        # 5) (옵션) 손쉬운 loss 계산
        loss = None
        if not prediction_loss_only and self.label_names:
            loss = torch.zeros(1, device=self.accelerator.device)

        # 6) Trainer 규격 반환
        return (loss, logits, None)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()

        # resume checkpoint 있을 경우 skip 적용
        resume_ckpt = getattr(self.args, "resume_from_checkpoint", None)
        if resume_ckpt and os.path.exists(os.path.join(resume_ckpt, "trainer_state.json")):
            with open(os.path.join(resume_ckpt, "trainer_state.json"), "r") as f:
                state = json.load(f)
            global_step = state.get("global_step", 0)
            per_device_bs = self.args.per_device_train_batch_size
            grad_accum = self.args.gradient_accumulation_steps
            world_size = self.args.world_size if hasattr(self.args, "world_size") else 1
            skip_samples = global_step * per_device_bs * grad_accum * world_size
            print(f"[RESUME] Skipping {skip_samples} samples")

            # SkipSampler 적용
            dataloader.sampler = SkipSampler(dataloader.sampler, skip_count=skip_samples)

        return dataloader