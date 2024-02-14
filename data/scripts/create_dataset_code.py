from datasets import load_dataset
import random
import json
import os
from tqdm import tqdm

# update this with your own token
HF_TOKEN = "hf_..."

# Load the dataset
code_dataset = load_dataset(
    "bigcode/the-stack-smol",
    split="train",
    token=HF_TOKEN,
)

subset = {}
for sample in tqdm(code_dataset, total=300_000):
    lang = sample["lang"]
    sample["byte_size"] = len(sample["content"].encode("utf-8"))
    sample["char_size"] = len(sample["content"])
    if lang not in subset:
        subset[lang] = []
    subset[lang].append(sample)

for lang in subset.keys():
    print(f"Found {len(subset[lang])} samples for {lang}")
    random.shuffle(subset[lang])
    # make sure to create the directories first
    os.makedirs("../code/test", exist_ok=True)
    with open(f"../code/test/{lang}.jsonl", "w") as f:
        for sample in subset[lang][:1000]:
            json.dump(sample, f)
            f.write("\n")
    os.makedirs("../code/train", exist_ok=True)
    with open(f"../code/train/{lang}.jsonl", "w") as f:
        for sample in subset[lang][1000:]:
            json.dump(sample, f)
            f.write("\n")
