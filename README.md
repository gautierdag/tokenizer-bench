# Code for the Paper: Getting the most out of your tokenizer for pre-training and domain adaptation

Authors: Gautier Dagan, Gabriele Synnaeve, Baptiste Rozière

## Installation

Install dependencies to a local python environment:

```bash
pip install -r requirements.txt
```

## Data

Download the data using the `datasets` library and the provided scrips in `data/scripts`. Make sure to update the HF_TOKEN variable to your own HF token.

## Training a tokenizer

The `train.py` script provides an minimal example to train a BPE tokenizer using the Huggingface library.

### Requirements

Training requires `tokenizers>="0.14.0"` since that version implements limiting the number of characters allowed per token.

## Training

```python
    output_dir: str

    english_datasets: str = "data/english/train"
    code_datasets: str = "data/code/train"
    multilingual_datasets: str = "data/multilingual/train"

    num_characters: int = 1_000_000_000
    vocab_size: int = 64_000
    max_sentencepiece_length: int = 128
    normalization_rule_name: str = "gpt"
    code_percentage: float = 0.1
    multilingual_percentage: float = 0.1

    output_name: Optional[str] = None
```

- `english_datasets`: path to english datasets to use. List using `,`. The files are expected to be jsonl files in the format as those found in our shuffled datasets.
- `code_datasets`: path to code datasets to use. List using `,`. The files are expected to be jsonl files in the format as those found in our shuffled datasets.
- `multilingual_datasets`: path to multilingual datasets to use. List using `,`. The files are expected to be jsonl files in the format as those found in our shuffled datasets.
- `code_percentage`: how much of the data seen by tokenizer should be code data (default 10%).
- `multilingual_percentage`: how much of the data seen by tokenizer should be multilingual data (default 10%).

- `num_characters`: how many characters will be passed to train the BPE tokenizer (recommended range 1B / 10B). Increasing this parameter normally leads to better tokenizers. If you increase it too high, the tokenizer library will be at risk of **silent** int overflows, which will cause the tokenizer to be sub-optimal.
- `vocab_size`: the number of tokens in the vocabulary
- `max_sentencepiece_length`: the maximum length (in characters) of an individual token. Default is 128 (as GPT4's tokenizer). This parameter only brings marginal improvements to compression, since most tokens will remain small. Keep this small, if you have a lot of duplicate files in your dataset.
- `normalization_rule_name`: the regex to apply during pre-tokenization. Supported values are `[gpt, gpt-num2, punct, punct-num2, identity]`. `gpt` is the GPT pre-tokenization regex, and `punct` is a similar regex that forbids tokens from mixing punctuations such as `.` with letters. The `-num2` versions limit the max digit length to 2 (instead of the default 3). `num2` should be used for small vocabularies (N < 50k) otherwise, the whole range of number 1-999 will not be covered. `identity` skips pre-tokenization entirely, and should be used with care (decrease `max_sentencepiece_length`) since it maximises compression but can cause entire phrases to be tokenized as a single token. This means that when decoding one should use `token_healing=True`.

### example command

```python
python -m scripts.tokenizers.train train --output-dir /path/to/tokenizer/folder --num-characters 100000 --vocab-size 5000
```

### Output

The output of the `train.py` script is a json file that describes the tokenizer class.

By default, all tokenizer have the same special tokens (idxs 0,1,2,3,4,5) and all have byte-fallback:

```python
"bos_piece": "<|begin_of_text|>"
"eos_piece": "<|end_of_text|>"
"pad_piece": "<pad>"
"fim_prefix": "<|fim_prefix|>"
"fim_middle": "<|fim_middle|>"
"fim_suffix": "<|fim_suffix|>"
```

### Usage

```python
from transformers import PreTrainedTokenizerFast

path = ".../32k_gpt.json"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=path,clean_up_tokenization_spaces=False)

text = "hello world"

# encoding:
tokens = tokenizer.encode(text)

# decoding:
tokenizer.decode(tokens, skip_special_tokens=False)

# pad token
n_words = len(tokenizer)
bos_id = tokenizer.vocab["<|begin_of_text|>"]
eos_id = tokenizer.vocab["<|end_of_text|>"]
pad_id = tokenizer.vocab["<pad>"]
```

## Running Evaluations

To run evaluations on tokenizers in the  `tokenizers` directory, you can run the evaluation script in `eval.py`. `eval.py` saves the results of each tokenizer as a json file which can then be parsed by the `process_evals.ipynb` notebook to generate the data from the paper.
You should be able to run `eval.py` out of the box on the tokenizers provided (same as in the paper).

You might need to modify file paths in `eval.py` to match your setup, if you wish to run evaluations on your own tokenizers.

## Alternate Tokenizer library

Because of certain limitations with the Huggingface tokenizer library, we also provide an bare-bones tokenizer trainer for regex tokenizers here: [bpeasy](https://github.com/gautierdag/bpeasy).

Note `bpeasy` was not used or evaluated in the paper, but was made separately to offer a more opinionated and minimalistic alternative to Huggingface's tokenizer. It only supports regex BPE tokenization, but can export the tokenizer file to a Huggingface format or tiktoken format (faster at encoding/decoding).

## Citation

```bibtex
@article{dagan2024getting,
  title={Getting the most out of your tokenizer for pre-training and domain adaptation},
  author={Dagan, Gautier and Synnaeve, Gabriele and Rozi{\`e}re, Baptiste},
  journal={arXiv preprint arXiv:2402.01035},
  year={2024}
}
```
