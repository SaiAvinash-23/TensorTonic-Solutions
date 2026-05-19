#%%
import numpy as np
from typing import List, Dict

#%%
class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[id, str] = {}
        self.vocab_size: int = 0

        # Special tokens
        self.pad_token: str = "<PAD>"
        self.unk_token: str = "<UNK>"
        self.bos_token: str = "<BOS>"
        self.eos_token: str = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        for special in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            self.word_to_id[special] = self.vocab_size
            self.id_to_word[self.vocab_size] = special
            self.vocab_size += 1

        unique_words = sorted(set(
            word for text in texts for word in text.lower().split()
        ))

        for word in unique_words:
            self.word_to_id[word] = self.vocab_size
            self.id_to_word[self.vocab_size] = word
            self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        unk_token = self.word_to_id[self.unk_token]
        # tokens = [self.word_to_id[self.bos_token]]
        tokens = [self.word_to_id.get(word, unk_token) for word in text.lower().split()]
        # tokens.append(self.word_to_id[self.eos_token])
        return tokens

    def decode(self, ids: List[int]) -> str:
        special = {self.word_to_id[t] for t in [self.pad_token, self.bos_token, self.eos_token]}

        return " ".join(
            self.id_to_word.get(id, self.unk_token)
            for id in ids if id not in special
        )

