from datasets import load_dataset
import itertools
import json
import os
import random

HF_TOKEN = "hf_..."


languages = [
    "it",
    "bg",
    "arz",
    "es",
    "sr",
    "de",
    "uk",
    "ar",
    "cs",
    "ca",
    "sl",
    "fa",
    "pl",
    "ru",
    "fi",
    "sv",
    "pt",
    "fr",
    "ja",
    "hr",
    "zh",
    "ro",
    "nl",
    "id",
    "hu",
    "nb",
    "ko",
    "da",
    "vi",
]
# Extract 11,000 samples for each language
num_samples = 11_000
subset = {}
for lang in languages:
    print(lang)
    # Load the dataset
    try:
        dataset = load_dataset(
            "graelo/wikipedia",
            streaming=True,
            split="train",
            token=HF_TOKEN,
            name=f"20230901.{lang}",
        )
    except:
        print(f"Failed to load {lang}")
        continue
    print(f"20230901.{lang}")
    subset[lang] = []
    for sample in itertools.islice(dataset, num_samples):
        sample["byte_size"] = len(sample["text"].encode("utf-8"))
        sample["char_size"] = len(sample["text"])
        subset[lang].append(sample)

    random.shuffle(subset[lang])
    print(f"Found {len(subset[lang])} samples")
    # dump to jsonl file
    # make sure to create the directories first
    os.makedirs("../multilingual/test", exist_ok=True)
    with open(f"../multilingual/test/{lang}.jsonl", "w") as f:
        for sample in subset[lang][:1000]:
            json.dump(sample, f)
            f.write("\n")
    os.makedirs("../multilingual/train", exist_ok=True)
    with open(f"../multilingual/train/{lang}.jsonl", "w") as f:
        for sample in subset[lang][1000:]:
            json.dump(sample, f)
            f.write("\n")
