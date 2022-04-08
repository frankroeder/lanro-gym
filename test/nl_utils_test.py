import numpy as np
from lanro.language_utils import parse_instructions, create_commands, Vocabulary
from lanro.utils import SHAPES, RGBCOLORS


def test_parse_instruction():
    instruction_list = [
        "Hello world",
        "World helo",
        "Sunny weather",
        "Lorem ipsum dolor sit amet",
    ]
    word_list, max_instr_len = parse_instructions(instruction_list)
    assert len(word_list) == 10
    assert max_instr_len == 5


def test_command_creation():
    commands = create_commands(["pick", "tick"], (RGBCOLORS.RED, SHAPES.SQUARE))
    assert len(commands) == 2


def test_command_creation_list():
    commands = np.concatenate([create_commands(["pick", "tick"], (color, SHAPES.SQUARE)) for color in RGBCOLORS])
    assert len(commands) == 24


def test_vocabulary():
    vocab = Vocabulary(["hello", "world", "sunny", "weather"])
    assert vocab("hello") == 1
    assert vocab.idx_to_word(0) == "<pad>"
    assert vocab.word_to_idx("hello") == 1
    assert vocab.word_to_idx("sunny") == 2
    assert len(vocab) == 5
