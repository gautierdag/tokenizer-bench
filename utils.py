import logging
import sys
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

TEST_PYTHON_FUNC = "import numpy as np \n\ndef add(a:int,b:int):\n  \treturn a + b\n\n# run code\nadd(1,2)\ndef sub(a:int,b:int):\n    return (a-b)\n "
TEST_JS_FUNC = """],\n\t.site .site"""
TEST_CUSTOM = "<filename>"
TEST_WEIRD = "".join([chr(i) for i in range(50, 150)]) + "".join(
    [chr(i) for i in range(200, 300)]
)
TEST_NT = ("\t" * 5) + ("\n" * 5) + ("\t" * 5)

VALIDATION_TESTS = [
    TEST_PYTHON_FUNC,
    TEST_JS_FUNC,
    TEST_CUSTOM,
    TEST_WEIRD,
    TEST_NT,
]


def decode(tokenizer, encoded: str) -> str:
    if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        return tokenizer.decode(
            encoded, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    else:
        return tokenizer.decode(encoded)


def encode(tokenizer, text: str, gpt=False) -> list[int]:
    if len(text) == 0:
        return []
    if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        return tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.encode(text)


def validate_tokenizer(tokenizer_name: str, tokenizer, verbose: bool = False) -> bool:
    # skip validation for these tokenizers
    if "byt5" in tokenizer_name or "deepseek" in tokenizer_name:
        return True
    for test in VALIDATION_TESTS:
        encoded = encode(tokenizer, test, gpt="_gpt_" in tokenizer_name)
        decoded = decode(tokenizer, encoded)
        if decoded != test:
            if verbose:
                print(f"{tokenizer} fails test:")
                print([tokenizer.decode([e]) for e in encoded])
                print(decoded)
            return False
    return True


def calc_vocab_overlap(tokenizer_vocab: set) -> float:
    """
    Calculate overlap of token vocab with itself
    """
    sorted_vocab = sorted(tokenizer_vocab, key=len)
    c = 0
    for i, token in tqdm(enumerate(sorted_vocab), total=len(tokenizer_vocab)):
        for t in sorted_vocab[i + 1 :]:
            if token in t:
                c += 1
    return c / len(tokenizer_vocab)


def calc_vocab_avg_token_length(tokenizer_vocab: set) -> float:
    """
    Calculate average length of tokens
    """
    return sum([len(t) for t in tokenizer_vocab]) / len(tokenizer_vocab)


def default_inner():
    return {"vocab_counter": Counter(), "lengths": []}


def default_outer():
    return defaultdict(default_inner)


def get_prob_distribution_counter(counter: Counter):
    # modified from https://github.com/zouharvi/tokenization-scorer/blob/main/tokenization_scorer/metrics.py
    words_freqs = list(counter.most_common())
    total_subwords = sum([x[1] for x in words_freqs])
    freqs = [freq for _, freq in words_freqs]
    probs = [freq / total_subwords for freq in freqs]
    vocab_size = len(words_freqs)
    return freqs, probs, vocab_size


def shannon_efficiency(counter: Counter):
    _, word_probs, vocab_size = get_prob_distribution_counter(counter)
    return -np.sum(word_probs * np.log2(word_probs)) / np.log2(vocab_size)


def renyi_efficiency(counter: Counter, power=3.0):
    if power == 1.0:
        return shannon_efficiency(counter)
    _, word_probs, vocab_size = get_prob_distribution_counter(counter)
    assert vocab_size > 0
    scale = 1 / (1 - power)
    val = (
        scale
        * np.log2(np.sum([prob**power for prob in word_probs]))
        / np.log2(vocab_size)
    )
    return val


def get_regex_from_normalization_rule_name(normalization_rule_name: str) -> str:
    # GPT4 regex
    if normalization_rule_name == "gpt":
        return r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    # limits to 2 digits (use for vocab size < 50k to ensure full digit coverage)
    elif normalization_rule_name == "gpt-num2":
        return r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    # separates punctuation from words (except spaces)
    elif normalization_rule_name == "punct":
        return r""" ?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    # limits to 2 digits (use for vocab size < 50k to ensure full digit coverage)
    elif normalization_rule_name == "punct-num2":
        return r""" ?\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    else:
        raise ValueError(f"Unknown normalization_rule_name {normalization_rule_name}")
