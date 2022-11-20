import numpy as np
from lanro_gym.env_utils import RGBCOLORS, SHAPES, TaskObject, valid_task_object_combination, dummys_not_goal_props
from lanro_gym.env_utils.object_properties import WEIGHTS
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.utils import goal_distance, scale_rgb, get_one_hot_list, get_prop_combinations, expand_enums, get_random_enum_with_exceptions


def test_get_prop_combinations():
    stream = [RGBCOLORS.RED, SHAPES.CYLINDER]
    combinations = get_prop_combinations(stream)
    assert len(combinations) == 2

    stream = [RGBCOLORS.RED, SHAPES.CYLINDER, RGBCOLORS.GREEN]
    combinations = get_prop_combinations(stream)
    assert len(combinations) == 4

    stream = [RGBCOLORS.RED, SHAPES.CYLINDER, RGBCOLORS.GREEN, SHAPES.CUBE]
    combinations = get_prop_combinations(stream)
    assert len(combinations) == 8


def test_expand_enums():
    expanded_enums = expand_enums([RGBCOLORS])
    assert len(expanded_enums) == 12
    expanded_enums = expand_enums([SHAPES])
    assert len(expanded_enums) == 3
    expanded_enums = expand_enums([WEIGHTS])
    assert len(expanded_enums) == 2
    expanded_enums = expand_enums([RGBCOLORS, SHAPES])
    assert len(expanded_enums) == 15
    expanded_enums = expand_enums([RGBCOLORS, SHAPES, WEIGHTS])
    assert len(expanded_enums) == 17


def test_get_random_enum_with_exceptions():
    assert get_random_enum_with_exceptions(RGBCOLORS, [RGBCOLORS.RED])[0] != RGBCOLORS.RED
    assert get_random_enum_with_exceptions(SHAPES, [SHAPES.CUBE])[0] != SHAPES.CUBE
    assert get_random_enum_with_exceptions(WEIGHTS, [WEIGHTS.HEAVY])[0] != WEIGHTS.HEAVY


def test_scale_rgb():
    assert np.allclose(scale_rgb([255., 255., 255.]), [1, 1, 1])
    assert np.allclose(scale_rgb([128., 128., 128.]), [.5019607, .5019607, .5019607])
    assert np.allclose(scale_rgb([0., 0., 0.]), [0, 0, 0])


def test_get_one_hot_list():
    one_hots = get_one_hot_list(1)
    assert len(one_hots) == 1
    assert np.all(one_hots[-1] == np.array([1.]))

    one_hots = get_one_hot_list(2)
    assert len(one_hots) == 2
    assert np.all(one_hots[0] == np.array([1., 0.]))
    assert np.all(one_hots[1] == np.array([0., 1.]))

    one_hots = get_one_hot_list(10)
    assert len(one_hots) == 10
    assert np.all(one_hots[-1] == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))


def test_goal_distance():
    vec1 = np.array([1, 1, 1])
    vec2 = np.array([2, 1.5, 2])
    assert goal_distance(vec1, vec2) == np.linalg.norm(vec1 - vec2)
    assert goal_distance(vec1, vec1) == 0
    assert goal_distance(vec2, vec2) == 0
    assert goal_distance(vec1, vec2) == 1.5


def test_valid_task_combinations():
    sim = PyBulletSimulation()
    task_obj1 = TaskObject(sim, RGBCOLORS.RED, SHAPES.CUBE)
    task_obj2 = TaskObject(sim, SHAPES.CUBE, RGBCOLORS.RED)
    assert not valid_task_object_combination(task_obj1, task_obj2)
    assert not valid_task_object_combination(task_obj2, task_obj1)

    task_obj3 = TaskObject(sim, RGBCOLORS.BLUE, SHAPES.CUBE)
    assert valid_task_object_combination(task_obj1, task_obj3)
    assert valid_task_object_combination(task_obj3, task_obj1)

    task_obj4 = TaskObject(sim, SHAPES.CUBE, RGBCOLORS.BLUE)
    assert valid_task_object_combination(task_obj2, task_obj3)
    assert valid_task_object_combination(task_obj3, task_obj2)

    task_obj5 = TaskObject(sim, RGBCOLORS.BLUE)
    assert not valid_task_object_combination(task_obj5, task_obj3)
    # valid, as we refer to the goal as "cube blue" when other object is not a cube
    if task_obj5.get_shape() != SHAPES.CUBE:
        assert valid_task_object_combination(task_obj3, task_obj5)
    assert not valid_task_object_combination(task_obj5, task_obj4)

    # valid, as we refer to the goal as "cube blue" when other object is not a cube
    if task_obj5.get_shape() != SHAPES.CUBE:
        assert valid_task_object_combination(task_obj4, task_obj5)

    task_obj6 = TaskObject(sim, RGBCOLORS.YELLOW)
    task_obj7 = TaskObject(sim, RGBCOLORS.BLUE)
    assert valid_task_object_combination(task_obj6, task_obj7)

    task_obj8 = TaskObject(sim, RGBCOLORS.YELLOW, SHAPES.CUBE)
    task_obj9 = TaskObject(sim, RGBCOLORS.YELLOW, SHAPES.CUBOID)
    assert valid_task_object_combination(task_obj8, task_obj9)

    task_obj10 = TaskObject(sim, SHAPES.CUBOID, RGBCOLORS.YELLOW)
    assert valid_task_object_combination(task_obj10, task_obj8)
    assert not valid_task_object_combination(task_obj10, task_obj9)


def test_valid_task_combinations2():
    sim = PyBulletSimulation()
    task_obj1 = TaskObject(sim, RGBCOLORS.RED)
    task_obj1._shape = SHAPES.CUBE
    assert task_obj1.has_dummy_weight and task_obj1.has_dummy_size

    task_obj2 = TaskObject(sim, SHAPES.CUBE)
    task_obj2.color = RGBCOLORS.RED
    assert task_obj2.has_dummy_weight and task_obj2.has_dummy_size
    # instruction: ... red object
    # red object (dummy == cube) and cube object (dummy == red)
    assert not valid_task_object_combination(task_obj1, task_obj2)

    task_obj3 = TaskObject(sim, SHAPES.CUBE)
    task_obj3.color = RGBCOLORS.BLUE
    # instruction: ... red object
    # red object (dummy == cube) and cube object (dummy == blue)
    assert valid_task_object_combination(task_obj1, task_obj3)

    task_obj4 = TaskObject(sim, SHAPES.CUBE)
    task_obj4.color = RGBCOLORS.BLUE
    assert not valid_task_object_combination(task_obj3, task_obj4)

    task_obj5 = TaskObject(sim, SHAPES.CUBOID)
    task_obj5.color = RGBCOLORS.BLUE
    assert valid_task_object_combination(task_obj4, task_obj5)

    task_obj6 = TaskObject(sim, WEIGHTS.HEAVY)
    task_obj6.color = RGBCOLORS.BLUE
    assert valid_task_object_combination(task_obj5, task_obj6)

    task_obj7 = TaskObject(sim, WEIGHTS.HEAVY)
    task_obj7.color = RGBCOLORS.BLUE
    assert not valid_task_object_combination(task_obj6, task_obj7)


def test_dummys_not_goal_primary():
    sim = PyBulletSimulation()
    task_obj1 = TaskObject(sim, primary=RGBCOLORS.RED, secondary=SHAPES.CUBE, onehot_idx=0)
    task_obj2 = TaskObject(sim, primary=RGBCOLORS.RED, onehot_idx=0)
    if task_obj2.get_shape() in [SHAPES.CUBE]:
        # 0: red cube and red object (dummy in {cube})
        assert not dummys_not_goal_props(task_obj1, task_obj2)
    else:
        # 1: red cube and red object (dummy in {cuboid, cylinder})
        assert dummys_not_goal_props(task_obj1, task_obj2)

    task_obj3 = TaskObject(sim, primary=RGBCOLORS.BLUE, onehot_idx=0)
    # 1: red cube and blue dummy
    assert dummys_not_goal_props(task_obj1, task_obj3)

    task_obj4 = TaskObject(sim, primary=SHAPES.CUBE, onehot_idx=0)
    if task_obj4.get_color() in [RGBCOLORS.RED]:
        assert not dummys_not_goal_props(task_obj1, task_obj4)
    else:
        # 1: red cube and cube object (dummy in {green, blue})
        assert dummys_not_goal_props(task_obj1, task_obj4)

    task_obj5 = TaskObject(sim, primary=SHAPES.CUBOID, onehot_idx=0)
    # 1: red cube and cuboid dummy
    assert dummys_not_goal_props(task_obj1, task_obj5)
