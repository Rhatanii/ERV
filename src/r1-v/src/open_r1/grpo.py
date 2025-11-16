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

from datasets import load_dataset, load_from_disk, Features,Sequence,Value
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, DataCollatorWithPadding

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, HumanOmniVLGRPOTrainer, HumanOmniVLGRPOVLLMTrainer
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
        default_factory=lambda: ["accuracy", "format"],
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
    "accuracy": accuracy_reward,
    "format": format_reward
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

from datasets import Dataset, DatasetDict
import random
random.seed(17)

def load_video_dataset(train_json_path: str, test_json_path: str, data_ratio: float = 1.0, test_data_ratio: float = 0.01):
    #### Train Dataset
    with open(train_json_path, 'r') as file:
        train_data = json.load(file)
    
    random.shuffle(train_data)
    train_data_length = int(len(train_data) * data_ratio)
    train_data = train_data[:train_data_length]
    
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

    trainer_cls = HumanOmniVLGRPOTrainer if not training_args.use_vllm else HumanOmniVLGRPOVLLMTrainer
    print("using: ", trainer_cls)

    # lora로 타겟할 위치 설정
    # target_module = []
    # for i in range(28):
    #     target_module.append(f'model.layers.{i}.self_attn.q_proj')
    #     target_module.append(f'model.layers.{i}.self_attn.k_proj')
    #     target_module.append(f'model.layers.{i}.self_attn.v_proj')
    #     target_module.append(f'model.layers.{i}.self_attn.o_proj')

    # model_args.lora_target_modules = target_module # ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
    
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
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
