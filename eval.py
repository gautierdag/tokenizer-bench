import json
import logging
import os
import sys

from collections import Counter
import glob

import numpy as np
import tiktoken
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from utils import validate_tokenizer, encode

os.environ["TIKTOKEN_CACHE_DIR"] = ""
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def get_content_key(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().rstrip()
        try:
            x = json.loads(line)
        except UnicodeDecodeError as e:
            print(f"Error when trying to decode '{line}': {str(e)}")
            raise
        for k in ["text", "content"]:
            if k in x:
                return k
        raise RuntimeError(f"Unable to determine key for {path}")


def get_datasets() -> dict:
    dataset = {}
    for path in glob.glob("data/*/test/*.jsonl"):
        name = path.split("/")[-1].replace(".jsonl", "")
        dataset[name] = {"path": path}
    return dataset


core_models = {
    "gpt2": "gpt2",
    "mpt": "mosaicml/mpt-7b-instruct",
    "bloom": "bigscience/bloom-7b1",
    "gpt-neox": "EleutherAI/gpt-neox-20b",
    "falcon": "tiiuae/falcon-40b",
    "pythia": "EleutherAI/pythia-12b",
    "codet5": "Salesforce/codet5-small",
    "incoder": "facebook/incoder-1B",
    "starcoder": "bigcode/starcoder",
    "replit": "replit/replit-code-v1_5-3b",
    "codegen": "Salesforce/codegen-350M-mono",
    "byt5": "google/byt5-small",
    "deepseek-coder": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "Yi-6B": "01-ai/Yi-6B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "santacoder": "bigcode/santacoder",
    "llama": "meta-llama/Llama-2-7b",
}


def get_tokenizer_name(model_path: str) -> str:
    tokenizer_name = model_path.split("/")[-1]
    if model_path.endswith(".json"):
        tokenizer_name = tokenizer_name.replace(".json", "")
    if model_path.endswith(".model"):
        tokenizer_name = tokenizer_name.replace(".model", "")
    return tokenizer_name


def load_tokenizer(model_path: str):
    tokenizer_name = get_tokenizer_name(model_path)
    if model_path.endswith(".json"):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=model_path)
    elif model_path.endswith(".model"):
        tokenizer = SentencePieceProcessor(model_file=model_path)
    elif model_path in core_models:
        tokenizer_name = model_path
        pretrained_name = core_models[model_path]
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_name, trust_remote_code=True
        )
    elif model_path == "gpt4":
        tokenizer = tiktoken.encoding_for_model("gpt-4")
        tokenizer_name = "gpt4"
    elif model_path == "llama":
        tokenizer = SentencePieceProcessor(model_file="tokenizers/llama.model")
        tokenizer_name = "llama"
    else:
        raise RuntimeError(f"Unknown model type: {model_path}")
    return tokenizer, tokenizer_name


def eval(
    model_path: str,
    eval_dir: str,
    max_sample_size=1000,
):
    tokenizer, tokenizer_name = load_tokenizer(model_path)
    assert validate_tokenizer(
        tokenizer_name, tokenizer, verbose=True
    ), f"{tokenizer_name} validation failed"
    logging.info(f"Evaluating model: {model_path}")
    datasets = get_datasets()

    data = {}
    for dataset_name in datasets.keys():
        dataset_path = datasets[dataset_name]["path"]
        print(f"Processing {dataset_name}: {dataset_path}")
        content_key = get_content_key(dataset_path)
        count = 0
        with open(dataset_path, "r") as f:
            data[dataset_name] = {"lengths": [], "vocab_counter": Counter()}
            for line in tqdm(f, total=max_sample_size):
                try:
                    text = json.loads(line)[content_key]
                except:
                    continue
                if text == "":
                    continue

                encoded_text = encode(tokenizer, text)
                data[dataset_name]["lengths"].append(len(encoded_text))
                if isinstance(encoded_text, np.ndarray):
                    encoded_text = encoded_text.tolist()
                data[dataset_name]["vocab_counter"].update(encoded_text)
                count += 1
                if count > max_sample_size:
                    break
    # save data
    out_path = f"{eval_dir}/{tokenizer_name}.eval.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"Saved data to {out_path}")


if __name__ == "__main__":
    to_eval = (
        list(core_models.keys())
        + glob.glob("tokenizers/*.json")
        + glob.glob("tokenizers/*.model")
    )
    print(f"Found {len(to_eval)} tokenizers to evaluate")
    for model_path in tqdm(to_eval):
        tokenizer, tokenizer_name = load_tokenizer(model_path)
        if os.path.exists(f"evals/{tokenizer_name}.eval.json"):
            print(f"Skipping {tokenizer_name} as it already exists")
            continue
        if not validate_tokenizer(tokenizer_name, tokenizer, verbose=True):
            print(f"Skipping {tokenizer_name} as it does not validate")
            continue
        eval(
            model_path,
            eval_dir="evals",
        )
