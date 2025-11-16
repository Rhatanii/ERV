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

import numpy as np
import torch
import torch.nn.functional as F

from datasets import load_dataset, load_from_disk, Features,Sequence,Value
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, BitsAndBytesConfig

from math_verify import parse, verify
from utils import  match_emotion_key_DFEW, match_emotion_key_MAFW, match_emotion_EMOBERT
from utils_reward import think_av_attention_reward, think_reward, think_step_reward, think_half_reward, think_top1_reward, think_with_hint_reward, think_confusion_reward, think_emotion_label_reward, think_step_with_neutral_reward, chain_reward, accuracy_reward, consistency_reward


from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, HumanOmniVLGRPOTrainer, HumanOmniVLGRPO_TEXT_Attention_Trainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config






@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "think", "chain", "consistency","think_with_hint", "think_emotion_label", "think_top1", "think_half","think_step", "think_step_with_neutral"],
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
    "think_av_attention": think_av_attention_reward,
    "think_step": think_step_reward,
    "think_half": think_half_reward,
    "chain": chain_reward,
    "accuracy": accuracy_reward,
    "format": format_reward,
    "consistency": consistency_reward,
    "think_with_hint": think_with_hint_reward,
    "think_confusion": think_confusion_reward,
    "think_emotion_label": think_emotion_label_reward,
    "think_top1": think_top1_reward,
    "think_step_with_neutral": think_step_with_neutral_reward,
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

    trainer_cls = HumanOmniVLGRPO_TEXT_Attention_Trainer 
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
