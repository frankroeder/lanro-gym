import pytest
import numpy as np
from lanro_gym.env_utils.object_properties import WEIGHTS
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.env_utils import TaskObject, RGBCOLORS, SHAPES, SIZES, DUMMY


def test_task_object_primary():
    sim = PyBulletSimulation()
    task_obj = TaskObject(sim, primary=RGBCOLORS.RED, onehot_idx=0)

    assert isinstance(task_obj.primary, RGBCOLORS)
    assert isinstance(task_obj.secondary, DUMMY)
    primary, _ = task_obj.get_properties()
    assert task_obj.primary == primary

    assert task_obj.get_color() == RGBCOLORS.RED
    assert task_obj.get_shape() == SHAPES.CUBE
    assert task_obj.get_size() == SIZES.MEDIUM
    assert task_obj.get_weight() == WEIGHTS.LIGHT

    task_obj_onehot = task_obj.get_onehot()
    assert len(task_obj_onehot) == 20
    assert np.all(
        task_obj_onehot == np.array([0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.]))
    assert task_obj.onehot_idx_colors == 0
    assert task_obj.obj_mass == 2


def test_task_object_primary2():
    sim = PyBulletSimulation()
    task_obj = TaskObject(sim, primary=SHAPES.CYLINDER, onehot_idx=2)
    assert isinstance(task_obj.primary, SHAPES)
    assert isinstance(task_obj.secondary, DUMMY)
    primary, _ = task_obj.get_properties()
    assert task_obj.primary == primary

    assert task_obj.get_color() == RGBCOLORS.RED
    assert task_obj.get_shape() == SHAPES.CYLINDER
    assert task_obj.get_size() == SIZES.MEDIUM
    assert task_obj.get_weight() == WEIGHTS.LIGHT

    task_obj_onehot = task_obj.get_onehot()
    assert len(task_obj_onehot) == 20
    assert np.all(
        task_obj_onehot == np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0.]))
    assert task_obj.onehot_idx_colors == 9
    assert task_obj.obj_mass == 2


def test_task_object_primary3():
    sim = PyBulletSimulation()
    task_obj = TaskObject(sim, primary=SHAPES.CYLINDER, onehot_idx=2, secondary=WEIGHTS.HEAVY, sec_onehot_idx=1)
    assert isinstance(task_obj.primary, SHAPES)
    assert isinstance(task_obj.secondary, WEIGHTS)
    primary, _ = task_obj.get_properties()
    assert task_obj.primary == primary

    assert task_obj.get_color() == RGBCOLORS.RED
    assert task_obj.get_shape() == SHAPES.CYLINDER
    assert task_obj.get_size() == SIZES.MEDIUM
    assert task_obj.get_weight() == WEIGHTS.HEAVY

    task_obj_onehot = task_obj.get_onehot()
    assert len(task_obj_onehot) == 20
    assert np.all(
        task_obj_onehot == np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1.]))
    assert task_obj.onehot_idx_colors == 9
    assert task_obj.obj_mass == 8


def test_task_object_with_secondary():
    sim = PyBulletSimulation()
    task_obj = TaskObject(sim, primary=RGBCOLORS.RED, secondary=SHAPES.CUBOID, onehot_idx=0, sec_onehot_idx=1)

    assert isinstance(task_obj.primary, RGBCOLORS)
    assert isinstance(task_obj.secondary, SHAPES)
    primary, secondary = task_obj.get_properties()
    assert task_obj.primary == primary
    assert task_obj.secondary == secondary

    assert task_obj.get_color() == RGBCOLORS.RED
    assert task_obj.get_shape() == SHAPES.CUBOID
    assert task_obj.get_size() == SIZES.MEDIUM
    assert task_obj.get_weight() == WEIGHTS.LIGHT

    task_obj_onehot = task_obj.get_onehot()
    assert len(task_obj_onehot) == 20
    assert np.all(
        task_obj_onehot == np.array([0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.]))
    assert task_obj.onehot_idx_colors == 0
    assert task_obj.obj_mass == 2


def test_task_object_with_secondary_error():
    sim = PyBulletSimulation()
    with pytest.raises(ValueError):
        TaskObject(sim, primary=SIZES.BIG, secondary=SIZES.BIG, onehot_idx=1, sec_onehot_idx=1)


def test_equality_objects():
    sim = PyBulletSimulation()
    task_obj1 = TaskObject(sim, primary=RGBCOLORS.RED, onehot_idx=1, secondary=SHAPES.CUBE)
    task_obj2 = TaskObject(sim, primary=RGBCOLORS.RED, onehot_idx=1, secondary=SHAPES.CUBE)
    assert task_obj1 == task_obj2

    task_obj3 = TaskObject(sim, primary=RGBCOLORS.BLUE, onehot_idx=1, secondary=SHAPES.CUBOID)
    assert task_obj1 != task_obj3

    task_obj4 = TaskObject(sim, primary=RGBCOLORS.RED, onehot_idx=1, secondary=SHAPES.CUBE)
    task_obj4._size = SIZES.BIG
    task_obj5 = TaskObject(sim, primary=RGBCOLORS.RED, onehot_idx=1, secondary=SHAPES.CUBE)
    task_obj5._size = SIZES.BIG
    assert task_obj4 == task_obj5

    task_obj6 = TaskObject(sim, primary=RGBCOLORS.RED, onehot_idx=1, secondary=SHAPES.CUBE)
    task_obj6.color = RGBCOLORS.BLUE
    task_obj6._size = SIZES.BIG
    task_obj7 = TaskObject(sim, primary=RGBCOLORS.RED, onehot_idx=1, secondary=SHAPES.CUBE)
    task_obj7._size = SIZES.SMALL
    assert task_obj6 != task_obj7

    task_obj8 = TaskObject(sim, primary=RGBCOLORS.BLUE, onehot_idx=1, secondary=WEIGHTS.HEAVY)
    assert task_obj8 != task_obj1
