from datasets import load_dataset
import itertools
import json
import os
import random

HF_TOKEN = "hf_..."

# Extract 10,000 samples for each language
num_samples = 11_000
c4_dataset = load_dataset(
    "c4",
    streaming=True,
    split="train",
    token=HF_TOKEN,
    name="en",
)
c4_samples = []
for sample in itertools.islice(c4_dataset, num_samples):
    sample["byte_size"] = len(sample["text"].encode("utf-8"))
    sample["char_size"] = len(sample["text"])
    c4_samples.append(sample)


random.shuffle(c4_samples)
print(f"Found {len(c4_samples)} samples")
# dump to jsonl file
# make sure to create the directories first
os.makedirs("../english/test", exist_ok=True)
with open("../english/test/c4.jsonl", "w") as f:
    for sample in c4_samples[:1000]:
        json.dump(sample, f)
        f.write("\n")
os.makedirs("../english/train", exist_ok=True)
with open("../english/train/c4.jsonl", "w") as f:
    for sample in c4_samples[1000:]:
        json.dump(sample, f)
        f.write("\n")


wiki_en_dataset = load_dataset(
    "graelo/wikipedia",
    streaming=True,
    split="train",
    token=HF_TOKEN,
    name="20230901.en",
)

wiki_en_samples = []
for sample in itertools.islice(wiki_en_dataset, num_samples):
    sample["byte_size"] = len(sample["text"].encode("utf-8"))
    sample["char_size"] = len(sample["text"])
    wiki_en_samples.append(sample)

random.shuffle(wiki_en_samples)
print(f"Found {len(wiki_en_samples)} samples")
# dump to jsonl file
# make sure to create the directories first
os.makedirs("../english/test", exist_ok=True)
with open("../english/test/wiki.jsonl", "w") as f:
    for sample in wiki_en_samples[:1000]:
        json.dump(sample, f)
        f.write("\n")
os.makedirs("../english/train", exist_ok=True)
with open("../english/train/wiki.jsonl", "w") as f:
    for sample in wiki_en_samples[1000:]:
        json.dump(sample, f)
        f.write("\n")
