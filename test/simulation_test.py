import numpy as np
from lanro.simulation import PyBulletSimulation


def test_init_step_close():
    sim = PyBulletSimulation()
    sim.step()
    sim.close()


def test_box_base_pos_orn():
    sim = PyBulletSimulation()
    body_name = "test_box"
    sim.create_box(body_name, [0.5, 0.5, 0.5], 1.0, [0, 0, 0], [1, 0, 0, 0])
    base_pos = sim.get_base_position(body_name)
    base_orn = sim.get_base_orientation(body_name)
    assert base_pos == (0, 0, 0)
    assert base_orn == (0, 0, 0, 1)
    assert sim._bodies_idx[body_name] == sim.get_object_id(body_name)
    sim.close()


def test_cylinder_base_pos_orn():
    sim = PyBulletSimulation()
    body_name = "test_cylinder"
    sim.create_cylinder(body_name, 0.5, 0.5, 1.0, [0, 0, 0], [1, 0, 0, 1])
    base_pos = sim.get_base_position(body_name)
    assert base_pos == (0, 0, 0)
    assert sim._bodies_idx[body_name] == sim.get_object_id(body_name)
    sim.close()


def test_sphere_base_pos_orn():
    sim = PyBulletSimulation()
    body_name = "test_sphere"
    sim.create_sphere(body_name, 0.5, 1.0, [0, 0, 0], [1, 0, 0, 1])
    base_pos = sim.get_base_position(body_name)
    assert base_pos == (0, 0, 0)
    assert sim._bodies_idx[body_name] == sim.get_object_id(body_name)
    sim.close()


def test_delta_t():
    sim = PyBulletSimulation()
    assert sim.dt == 1 / 500. * 20


def test_euler_quat():
    sim = PyBulletSimulation()
    quat = [0, np.pi, 0, 0]
    assert sim.get_euler_from_quaternion(quat) == (3.141592653589793, -0.0, 3.141592653589793)
    euler = [0, np.pi, 0]
    assert sim.get_quaternion_from_euler(euler) == (0.0, 1.0, 0.0, 6.123233995736766e-17)
