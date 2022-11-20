import numpy as np
from lanro_gym.language_utils import parse_instructions, create_commands, Vocabulary, word_in_string
from lanro_gym.env_utils import SHAPES, RGBCOLORS


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
    commands = create_commands("instruction", (RGBCOLORS.RED, SHAPES.CUBE), action_verbs=["pick", "tick"])
    assert len(commands) == 2


def test_command_creation_synonyms():
    commands = create_commands("instruction", (RGBCOLORS.RED, SHAPES.CUBE),
                               action_verbs=["pick", "tick"],
                               use_synonyms=True)
    assert len(commands) == 12


def test_command_creation_synonyms_only():
    commands = create_commands("instruction", (RGBCOLORS.RED, SHAPES.CUBE),
                               action_verbs=["pick", "tick"],
                               use_base=False,
                               use_synonyms=True)
    assert len(commands) == 4


def test_command_creation_list():
    commands = np.concatenate(
        [create_commands("instruction", (color, SHAPES.CUBE), action_verbs=["pick", "tick"]) for color in RGBCOLORS])
    assert len(commands) == 24


def test_command_creation_list_synonyms():
    commands = np.concatenate([
        create_commands("instruction", (color, SHAPES.CUBE), action_verbs=["pick", "tick"], use_synonyms=True)
        for color in RGBCOLORS
    ])
    assert len(commands) == 144


def test_command_creation_list_synonyms_only():
    commands = np.concatenate([
        create_commands("instruction", (color, SHAPES.CUBE),
                        action_verbs=["pick", "tick"],
                        use_base=False,
                        use_synonyms=True) for color in RGBCOLORS
    ])
    assert len(commands) == 48


def test_action_repair_command_creation():
    commands = create_commands("repair", (RGBCOLORS.RED, SHAPES.CUBE))
    assert len(commands) == 6


def test_action_repair_command_creation_synonyms():
    commands = create_commands("repair", (RGBCOLORS.RED, SHAPES.CUBE), use_synonyms=True)
    assert len(commands) == 36


def test_action_repair_command_creation_synonyms_only():
    commands = create_commands("repair", (RGBCOLORS.RED, SHAPES.CUBE), use_base=False, use_synonyms=True)
    assert len(commands) == 12


def test_negation_command_creation():
    commands = create_commands("negation", (RGBCOLORS.RED, SHAPES.CUBE))
    assert len(commands) == 1


def test_negation_command_creation_synonyms():
    commands = create_commands("negation", (RGBCOLORS.RED, SHAPES.CUBE), use_synonyms=True)
    assert len(commands) == 6


def test_negation_command_creation_synonyms_only():
    commands = create_commands("negation", (RGBCOLORS.RED, SHAPES.CUBE), use_base=False, use_synonyms=True)
    assert len(commands) == 2


def test_word_in_string():
    instruction_string = "Lorem ipsum dolor sit amet"
    search_words = np.array(["ipsum"])
    assert "ipsum" == word_in_string(instruction_string, search_words)
    assert "" == word_in_string(instruction_string, np.array(["dog", "cat"]))


def test_vocabulary():
    vocab = Vocabulary(["hello", "world", "sunny", "weather"])
    assert vocab("hello") == 1
    assert vocab.idx_to_word(0) == "<pad>"
    assert vocab.word_to_idx("hello") == 1
    assert vocab.word_to_idx("sunny") == 2
    assert len(vocab) == 5
