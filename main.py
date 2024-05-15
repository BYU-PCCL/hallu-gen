import argparse
import numpy as np
import torch
import pandas as pd

MODEL_NAMES = {
    'llama2_chat_7B': '../../models/llama2/hf/Llama-2-7b-chat-hf',  
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf',  
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
}

DATASET_NAMES = {
    'factCHD': 'factCHD.csv',
    'hallugen': 'path/to/hallugen'
}

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, default='llama2_chat_7B', choices=MODEL_NAMES.keys(), help='model name')
    parser.add_argument('--dataset', type=str, default='factCHD', choices=DATASET_NAMES.keys(), help='feature bank for training probes')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--ft_method', type=str, default='pyreft', help='finetuning method')
    args = parser.parse_args()
    

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv(DATASET_NAMES[args.dataset])

    

 

if __name__ == "__main__":
    main()