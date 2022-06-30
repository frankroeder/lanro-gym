import numpy as np
from lanro_gym.utils import TaskObject, TaskObjectList, RGBCOLORS, SHAPES, get_one_hot_list
from lanro_gym.simulation import PyBulletSimulation


def test_task_object():
    sim = PyBulletSimulation()
    color = RGBCOLORS.RED
    onehot_color = np.array([1, 0])
    shape = SHAPES.SQUARE
    onehot_shape = np.array([1, 0])
    obj = TaskObject(sim, color=color, onehot_color=onehot_color, shape=shape, onehot_shape=onehot_shape)
    assert obj.get_color() == color
    assert obj.get_shape() == shape
    primary_prop, secondary_prop = obj.get_properties()
    assert primary_prop == color
    assert secondary_prop == shape
    assert np.all(obj.get_onehot() == np.array([1, 0, 1, 0]))


def test_task_object_list_shape_mode():
    sim = PyBulletSimulation()
    obj_list = TaskObjectList(sim, shape_mode=True)
    assert len(obj_list) == 9
    assert obj_list[0].get_color() == RGBCOLORS.RED
    assert obj_list[2].get_color() == RGBCOLORS.RED
    assert np.all(obj_list[0].get_onehot() == np.array([1, 0, 0, 1, 0, 0]))
    assert np.all(obj_list[1].get_onehot() == np.array([1, 0, 0, 0, 1, 0]))


def test_task_object_list_color_mode():
    sim = PyBulletSimulation()
    obj_list = TaskObjectList(sim, color_mode=True)
    assert len(obj_list) == 9
    assert obj_list[1].get_color() == RGBCOLORS.GREEN
    assert obj_list[2].get_color() == RGBCOLORS.BLUE
    assert obj_list[3].get_color() == RGBCOLORS.YELLOW
    assert np.all(obj_list[1].get_onehot() == np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.]))
    assert np.all(obj_list[2].get_onehot() == np.array([0., 0., 1., 0., 0., 0., 0., 0., 0.]))
    assert np.all(obj_list[3].get_onehot() == np.array([0., 0., 0., 1., 0., 0., 0., 0., 0.]))

def test_task_object_list_colorshape_mode():
    sim = PyBulletSimulation()
    obj_list = TaskObjectList(sim, color_mode=True, shape_mode=True)
    assert len(obj_list) == 27
    assert obj_list[4].get_shape() == SHAPES.RECTANGLE
    assert obj_list[4].get_color() == RGBCOLORS.GREEN
    assert obj_list[5].get_shape() == SHAPES.CYLINDER
    assert obj_list[5].get_color() == RGBCOLORS.GREEN


def test_get_one_hot_list():
    one_hots = get_one_hot_list(10)
    assert len(one_hots) == 10
    assert np.all(one_hots[-1] == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
