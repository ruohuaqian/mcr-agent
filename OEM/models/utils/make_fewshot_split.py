# tools/make_fewshot_split.py
import json, argparse, random, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--k", type=int, required=True, help="few-shot K (e.g., 1 or 5)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--shuffle", action="store_true", help="shuffle before taking K")
    args = ap.parse_args()

    with open(args.infile) as f:
        data = json.load(f)

    # 找到训练列表（兼容不同命名）
    train_key = "train" if "train" in data else ("train_seen" if "train_seen" in data else None)
    if train_key is None:
        raise ValueError("No train/train_seen found in splits json")

    train = list(data[train_key])
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(train)

    k = max(1, min(args.k, len(train)))
    data[train_key] = train[:k]

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    with open(args.outfile, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[few-shot] {train_key}: {len(train)} -> {k}")
    print(f"[few-shot] wrote: {args.outfile}")

if __name__ == "__main__":
    main()
