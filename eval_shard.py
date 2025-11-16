'''
inference에서 data iteration.
+ sharding
+ logging


'''
import os
import json
import argparse
import yaml
from humanomni import model_init, mm_infer
from humanomni.utils import disable_torch_init
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from data import VideoDataset

os.environ['TRANSFORMERS_OFFLINE'] = '1'


def read_json(file_path):
    with open(file_path, 'r') as rf:
        data = json.load(rf)
    return data

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="HumanOmni Inference Script")
    parser.add_argument('--config', type=str, required=True, help='Path to the config.yaml file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--modal', type=str, required=True, help='modality ')
    parser.add_argument('--shards', type=int, default=4, help='shard Dataloader')
    parser.add_argument('--shard_id', type=int, default=0, help='shard ID')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output file')
    parser.add_argument('--model_path', type=str, required=True, help='model path name')
    parser.add_argument('--instruct', type=str, required=True, help='instruction')
    parser.add_argument('--use_AU', type=lambda x: x.lower() == "true", default=False, help='whether to use AU information with table')
    parser.add_argument('--use_AU01', type=lambda x: x.lower() == "true", default=False, help='whether to use AU information with table')
    parser.add_argument('--use_AU01_w_table', type=lambda x: x.lower() == "true", default=False, help='whether to use AU information with table')
    parser.add_argument('--use_AU01_w_info', type=lambda x: x.lower() == "true", default=False, help='whether to use AU information with emotion related info')
    parser.add_argument('--use_only_AU_matched_dataset', type=lambda x: x.lower() == "true", default=False, help='whether to use only AU matched dataset (for reasoning dataset generation)')
    parser.add_argument('--use_special_format', type=lambda x: x.lower() == "true", default=False, help='whether to use special format for output')
    parser.add_argument('--use_new_av_format', type=lambda x: x.lower() == "true", default=False, help='whether to use new audio-visual format')
    parser.add_argument('--do_sample', type=lambda x: x.lower() == "true", default=False, help='whether to do sampling')
    parser.add_argument('--temperature',type=float,default=0.5, help='temperature for sampling' )
    args = parser.parse_args()
    config = load_config(args.config)
    
    
    modal =args.modal
    model_path = args.model_path
    instruct = args.instruct
    use_AU = args.use_AU
    use_AU01 = args.use_AU01
    use_AU01_w_table = args.use_AU01_w_table
    use_AU01_w_info = args.use_AU01_w_info
    use_special_format = args.use_special_format
    use_new_av_format = args.use_new_av_format
    au_dict_path = "/mnt/ssd_hs/Exp/R1-Omni/data/AU/AU_match.json"
    au_information_dict = read_json(au_dict_path)
    

    video_dataset_name = config['data']['video_dataset_name']
    au_emotion_path = f"/mnt/ssd_hs/Exp/R1-Omni/data/AU/emotion_AU_{video_dataset_name}.json"
    # au_emotion_path = f"/mnt/ssd_hs/Exp/R1-Omni/data/AU/emotion_AU_MAFW.json"
    label_path = config['data']['label_path']
    
    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cuda_device = str(config.get('CUDA_VISIBLE_DEVICES', '0'))
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    

    bert_model = "bert-base-uncased"
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model)

    disable_torch_init()

    # Init
    model, processor, tokenizer = model_init(model_path,**config)

    # Load Dataset
    full_dataset = VideoDataset(video_dataset_name, label_path, processor, modal, use_only_AU_matched_dataset=args.use_only_AU_matched_dataset)
    if args.shards > 1:
        total = len(full_dataset)

        total = len(full_dataset)
        per_shard = (total + args.shards - 1) // args.shards  # ceil-div
        start = args.shard_id * per_shard
        end = min(start + per_shard, total)
        shard_indices = list(range(start, end))
        
        
        dataset = Subset(full_dataset, shard_indices)
        print(f"Running shard {args.shard_id+1}/{args.shards}: {len(shard_indices)} samples")
    else:
        dataset = full_dataset
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    os.makedirs('results', exist_ok=True)
    
    for batch in tqdm(dataloader, desc="Evaluating"):
            video_paths = batch['video_path']
            video_tensors = batch['video_tensor']
            if modal == 'video_audio' or modal == 'audio':
                audio_tensors = batch['audio_tensor']
            else:
                audio_tensors = None
            print('use_AU:',use_AU)
            print('use_AU01:',use_AU01)
            print('use_AU01_w_table:',use_AU01_w_table)
            print('use_AU01_w_info:',use_AU01_w_info)
            print('use_special_format:',use_special_format)
            print('use_new_av_format:',use_new_av_format)
            if use_AU:
                
                # AU or AU01
                if not use_AU01_w_table:
                    if use_AU01:
                        au_list = batch['au01_list']
                    else:
                        au_list = batch['au_list']
                    
                    au_info = []
                    for au in au_list:
                        au_info.append(au[0])

                    au_info = ', '.join(au_info)
                    print("AU info:",au_info)
                    if use_special_format:
                        instruct = f"<video>\n<audio>\nAs an emotion recognition expert, analyze the video considering facial AUs, visual behavior, and audio tone. Describe your observations using the tags <fau_desc>, <vis_desc>, and <aud_desc> for facial, visual, and audio cues respectively. You are also provided with information about facial Action Units (AUs) detected from the characters' expressions. If relevant, incorporate and reference these AU ({au_info})to support emotion inference. Then, integrate all these cues into a reasoning process in <think> </think>, and provide the final predicted emotion in <answer> </answer> tags."
                    else:
                        instruct = f"<video>\n<audio>\nAs an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you? You are also provided with information about facial Action Units (AUs) detected from the characters' expressions. In your reasoning process, if relevant, incorporate and reference these AU ({au_info}) to support your emotion inference. Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
                    
                # AU01 with table
                else:
                    if not use_AU01_w_info:
                            
                        if use_AU01:
                            au_list = batch['au01_list']
                        else:
                            au_list = batch['au_list']
                        
                        au_info = ''
                        for aus in au_list:
                            au_info += aus[0] +':'+ str(au_information_dict.get(aus[0], []))
                        # print("AU info with table:",au_info)
                        instruct = f"<video>\n<audio>\nAs an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you? You are also provided with information about facial Action Units (AUs) detected from the characters' expressions. In your reasoning process, if relevant, incorporate and reference these AU ({au_info}) to support your emotion inference. Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
                    
                    # AU 01 with table and emotion related info
                    else:
                        if use_AU01:
                            au_list = batch['au01_list']
                        else:
                            au_list = batch['au_list']
                        
                        au_info = ''
                        for aus in au_list:
                            au_info += aus[0] +':'+ str(au_information_dict.get(aus[0], []))
                        # print("AU info with table and emotion related info:",au_info)
                        
                        au_emotion_dict = read_json(au_emotion_path)
                        au_emotion_dict_str =str(au_emotion_dict)
                        
                        if not use_special_format:
                            instruct = f"<video>\n<audio>\nAs an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you? You are also provided with information about facial Action Units (AUs) detected from the characters' expressions. In your reasoning process, if relevant, incorporate and reference these AU ({au_info}) to support your emotion inference. Here is Emotion-AU Relation Table {au_emotion_dict_str}. Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
                        else:
                            instruct = f"""<video>\n<audio>\nAs an emotional recognition expert, throughout the video, which emotion conveyed by the characters is the most obvious to you? You are also provided with information about facial Action Units (AUs) detected from the characters' expressions. In your reasoning process, if relevant, incorporate and reference these AU ({au_info}) to support your emotion inference. Here is the Emotion-AU Relation Table {au_emotion_dict_str}. \n\nOutput the following elements in structured tags:\n- <fau_desc>Describe key AUs and their indicative emotional signals.</fau_desc>\n<vis_desc> Description based on visual cues (facial, posture, gestures) from the video </vis_desc>\n<aud_desc> Description based on audio cues from the video (tone, pitch, intensity, speech rate).</aud_desc>\n<think>The integrated reasoning process combining all modalities and AU evidence. </think>\n<answer>Final predicted emotion label.</answer>"""
            else:
                if use_new_av_format:
                    instruct ="<video>\n<audio>\nAs an emotion recognition expert, analyze the video considering visual behavior, and audio tone. Describe your observations using the tags <vis_desc> and <aud_desc> for visual, and audio cues respectively. Focus first on the modality that provides the most important information for emotion prediction. Then, integrate all these cues into a reasoning process in <think> </think>, and provide the final predicted emotion in <answer> </answer> tags."
                else:    
                    instruct = args.instruct

            
            print("Final instruction:",instruct)
            for i in range(len(video_paths)):
                output = mm_infer(
                    video_tensors[i],
                    instruct,
                    model=model,
                    tokenizer=tokenizer,
                    modal=modal,
                    question=instruct,
                    bert_tokeni=bert_tokenizer,
                    do_sample=args.do_sample,
                    
                    audio=audio_tensors[i] if audio_tensors is not None else None,
                    model_size=config['model']['model_size'],
                    temperature=args.temperature
                )

                video_name = os.path.splitext(os.path.basename(video_paths[i]))[0]
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write('|'.join([video_name,output])+'\n')

    
    


if __name__ == "__main__":
    main()
