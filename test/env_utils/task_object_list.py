import numpy as np
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.env_utils import TaskObjectList, RGBCOLORS, SHAPES


def test_task_object_list_default():
    sim = PyBulletSimulation()
    obj_list = TaskObjectList(sim)
    assert len(obj_list) == 3
    assert obj_list[0].get_color() == RGBCOLORS.RED
    assert obj_list[1].get_color() == RGBCOLORS.GREEN
    assert obj_list[2].get_color() == RGBCOLORS.BLUE

    task_obj_args = obj_list.get_task_obj_args({}, RGBCOLORS.RED, primary=True)
    assert task_obj_args['primary'] == RGBCOLORS.RED
    assert task_obj_args['onehot_idx'] == 9

    task_obj_args = obj_list.get_task_obj_args({}, RGBCOLORS.RED, primary=False)
    assert task_obj_args['secondary'] == RGBCOLORS.RED
    assert task_obj_args['sec_onehot_idx'] == 9


def test_task_object_list_shape_mode():
    sim = PyBulletSimulation()
    obj_list = TaskObjectList(sim, shape_mode=True)
    assert len(obj_list) == 24
    assert obj_list[0].get_color() == RGBCOLORS.RED
    assert obj_list[5].get_shape() == SHAPES.CYLINDER
    expected_oh = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0.])
    assert np.all(obj_list[0].get_onehot() == expected_oh)


def test_task_object_list_color_mode():
    sim = PyBulletSimulation()
    obj_list = TaskObjectList(sim, color_mode=True)
    assert len(obj_list) == 9
    obj_props = obj_list.get_obj_properties()
    assert len(obj_props) == 9


def test_task_object_list_weight_mode():
    sim = PyBulletSimulation()
    obj_list = TaskObjectList(sim, weight_mode=True)
    assert len(obj_list) == 17
    obj_props = obj_list.get_obj_properties()
    assert len(obj_props) == 17


def test_task_object_list_size_mode():
    sim = PyBulletSimulation()
    obj_list = TaskObjectList(sim, size_mode=True)
    assert len(obj_list) == 24
    obj_props = obj_list.get_obj_properties()
    assert len(obj_props) == 24


def test_task_object_list_sizeshape_mode():
    sim = PyBulletSimulation()
    obj_list = TaskObjectList(sim, shape_mode=True, size_mode=True)
    assert len(obj_list) == 63
    obj_props = obj_list.get_obj_properties()
    assert len(obj_props) == 63


def test_task_object_list_colorshape_mode():
    sim = PyBulletSimulation()
    obj_list = TaskObjectList(sim, color_mode=True, shape_mode=True)
    assert len(obj_list) == 66
    obj_props = obj_list.get_obj_properties()
    assert len(obj_props) == 66


def test_task_object_list_weightshape_mode():
    sim = PyBulletSimulation()
    obj_list = TaskObjectList(sim, weight_mode=True, shape_mode=True)
    assert len(obj_list) == 50
    obj_props = obj_list.get_obj_properties()
    assert len(obj_props) == 50


def test_task_object_list_colorshapesize_mode():
    sim = PyBulletSimulation()
    obj_list = TaskObjectList(sim, color_mode=True, shape_mode=True, size_mode=True)
    assert len(obj_list) == 141
    obj_props = obj_list.get_obj_properties()
    assert len(obj_props) == 141
