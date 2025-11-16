'''
inference에서 data iteration.
+ sharding
+ logging


'''
import os
import argparse
import yaml
from humanomni import model_init, mm_infer
from humanomni.utils import disable_torch_init
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader

from data import VideoDataset

os.environ['TRANSFORMERS_OFFLINE'] = '1'




def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="HumanOmni Inference Script")
    parser.add_argument('--config', type=str, required=True, help='Path to the config.yaml file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')

    args = parser.parse_args()
    config = load_config(args.config)
    
    
    modal =config.get('modal', 'video_audio')
    model_path = config['data']['model_path']
    video_dataset_name = config['data']['video_dataset_name']
    label_path = config['data']['label_path']
    instruct = config['instruct']
    output_path = config['output_path']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cuda_device = str(config.get('CUDA_VISIBLE_DEVICES', '0'))
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    

    bert_model = "bert-base-uncased"
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model)

    disable_torch_init()

    # Init
    model, processor, tokenizer = model_init(model_path,**config)

    # Load Dataset
    dataset = VideoDataset(video_dataset_name, label_path, processor, modal)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    os.makedirs('results', exist_ok=True)
    
    for batch in tqdm(dataloader, desc="Evaluating"):
            video_paths = batch['video_path']
            video_tensors = batch['video_tensor']
            if modal == 'video_audio' or modal == 'audio':
                audio_tensors = batch['audio_tensor']
            else:
                audio_tensors = None

            for i in range(len(video_paths)):
                output = mm_infer(
                    video_tensors[i],
                    instruct,
                    model=model,
                    tokenizer=tokenizer,
                    modal=modal,
                    question=instruct,
                    bert_tokeni=bert_tokenizer,
                    do_sample=False,
                    audio=audio_tensors[i] if audio_tensors is not None else None,
                    model_size=config['model']['model_size'],
                )

                video_name = os.path.splitext(os.path.basename(video_paths[i]))[0]
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write('|'.join([video_name,output])+'\n')

    
    


if __name__ == "__main__":
    main()
