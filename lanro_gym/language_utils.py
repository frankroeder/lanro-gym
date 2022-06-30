from typing import List, Set, Tuple
import numpy as np


def create_commands(action_verbs: List[str], property_tuple: Tuple) -> List[str]:
    sentences = []
    for verb in action_verbs:
        primary_property, secondary_property = property_tuple
        command = verb + " the " + primary_property.name.lower() + " " + secondary_property.name.lower()
        sentences.append(command)
    return sentences


def parse_instructions(instructions: List[str]) -> Tuple[Set[str], int]:
    word_list = []
    max_instruction_len = 0
    for _instrucion in instructions:
        _splitted = _instrucion.lower().split(' ')
        if len(_splitted) > max_instruction_len:
            max_instruction_len = len(_splitted)
        word_list.extend(_splitted)
    return set(word_list), max_instruction_len


class Vocabulary:

    def __init__(self, words: List[str]):
        word_list = ['<pad>'] + sorted(list(set(words)))
        _idx_list = np.arange(0, len(word_list))
        self.idx2word = dict(zip(_idx_list, word_list))
        self.word2idx = dict(zip(word_list, _idx_list))
        assert len(self.idx2word) == len(self.word2idx)

    def idx_to_word(self, idx: int) -> str:
        return self.idx2word[idx]

    def word_to_idx(self, word: str) -> int:
        return self.word2idx[word]

    def __call__(self, word) -> int:
        return self.word_to_idx(word)

    def __len__(self) -> int:
        return len(self.word2idx)
