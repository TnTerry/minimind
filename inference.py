import warnings
import argparse

import torch
from transformers import AutoTokenizer

from model.model import MiniMindLM
from model.LMConfig import LMConfig

warnings.filterwarnings('ignore')


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('/root/minimind/model/minimind_tokenizer')
    model = MiniMindLM(lm_config).to(args.device)
    print(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Inference")
    parser.add_argument("--model_path", type=str, default="/root/data/minimind_model/pretrain_512.pth")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    # 与LMConfig中的保持一致，也可以不设定
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    args = parser.parse_args()

    torch.manual_seed(1024)

    lm_config = LMConfig(
        dim=args.dim, n_layers=args.n_layers, 
        max_seq_len=args.max_seq_len, use_moe=args.use_moe
    )
    model, tokenizer = init_model(lm_config)