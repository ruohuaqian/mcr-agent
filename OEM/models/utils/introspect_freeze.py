#!/usr/bin/env python
import argparse, torch
from importlib import import_module
from collections import defaultdict

def topk(d, k=20):
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g., OEM.models.model.seq2seq_im_mask_obj")
    ap.add_argument("--ckpt", required=True, help="checkpoint path")
    ap.add_argument("--levels", type=int, default=1, help="prefix depth (1 or 2)")
    args = ap.parse_args()

    M = import_module(args.model)
    model, _ = M.Module.load(args.ckpt)  # 直接用仓库里的加载函数
    total = 0
    pref_count = defaultdict(int)
    pref_params = defaultdict(int)

    for n, p in model.named_parameters():
        numel = p.numel()
        total += numel
        parts = n.split(".")
        key = parts[0] if args.levels <= 1 else ".".join(parts[:min(args.levels, len(parts))])
        pref_count[key] += 1
        pref_params[key] += numel

    print(f"# total parameters: {total:,}")
    print(f"# top prefixes (levels={args.levels}) by param-count:")
    for k, v in topk(pref_params, 30):
        print(f"{k:40s}  params={v:>10,d}  tensors={pref_count[k]:>6d}")

if __name__ == "__main__":
    main()
