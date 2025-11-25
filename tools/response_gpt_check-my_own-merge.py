import json
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="R1-Omni additional GPT reasoning")
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset_name')
    parser.add_argument('--data_type', type=str, required=True, help='dataset type train ,test ')
    parser.add_argument('--nshard', type=int, default=1, help='number of shards')
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    parser.add_argument('--instruction_label_name', type=str, required=True, help='instruction dataset')

    args = parser.parse_args()
    file_names =[]
        
    for i in range(args.nshard):
        file_names.append(f"/mnt/ssd_hs/Exp/R1-Omni/results/{args.dataset_name}/{args.model_id}/label_false/output_eval5_all-video_audio-instruct_{args.instruction_label_name}-{i}.json")
    
    merged_data = []
    merged_output_file =f"/mnt/ssd_hs/Exp/R1-Omni/results/{args.dataset_name}/{args.model_id}/label_false/output_eval5_all-video_audio-instruct_{args.instruction_label_name}.json"
    
    for file_name in file_names:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_data.extend(data)  
            
    with open(merged_output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
