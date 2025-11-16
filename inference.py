import os
import argparse
import yaml
from humanomni import model_init, mm_infer
from humanomni.utils import disable_torch_init
from transformers import BertTokenizer

# 设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="HumanOmni Inference Script")
    parser.add_argument('--config', type=str, required=True,default='/mnt/ssd_hs/Exp/R1-Omni/configs/inference_text_only.yaml', help='Path to the config.yaml file')

    args = parser.parse_args()

    config = load_config(args.config)
    
    
    modal =config.get('modal', 'video_audio')
    model_path = config['data']['model_path']
    video_path = config['data']['video_path']
    instruct = config['instruct']
    cuda_device = str(config.get('CUDA_VISIBLE_DEVICES', '0'))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    
    # 初始化BERT分词器
    bert_model = "bert-base-uncased"
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model)

    # 禁用Torch初始化
    disable_torch_init()

    # 初始化模型、处理器和分词器
    model, processor, tokenizer = model_init(model_path,**config)
  #  import ipdb;ipdb.set_trace()

    # 处理视频输入
    video_tensor = processor['video'](video_path)
    
    # 根据modal类型决定是否处理音频
    if modal == 'video_audio' or modal == 'audio':
        audio = processor['audio'](video_path)[0]
    else:
        audio = None
    
    # for name, param in model.named_parameters():
    #     with open('./model_parameters.txt','a') as f:
    #         f.write(f'{name}: {param.dtype} {param.requires_grad}\n')
    print(instruct)
    output = mm_infer(video_tensor,
                      instruct, 
                      model=model, 
                      tokenizer=tokenizer, 
                      modal=modal, 
                      question=instruct, 
                      bert_tokeni=bert_tokenizer, 
                      do_sample=True, 
                      audio=audio,
                      model_size=config['model']['model_size'])
    print(output)
    with open('./output_human0.5B.txt','w') as f:
        f.write(output)

if __name__ == "__main__":
    main()
