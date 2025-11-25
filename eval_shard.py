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
    parser.add_argument('--do_sample', type=lambda x: x.lower() == "true", default=False, help='whether to do sampling')
    parser.add_argument('--temperature',type=float,default=0.5, help='temperature for sampling' )
    args = parser.parse_args()
    config = load_config(args.config)
    
    
    modal =args.modal
    model_path = args.model_path
    instruct = args.instruct
    video_dataset_name = config['data']['video_dataset_name']
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
    full_dataset = VideoDataset(video_dataset_name, label_path, processor, modal)
    if args.shards > 1:
        total = len(full_dataset)
        per_shard = (total + args.shards - 1) // args.shards 
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
