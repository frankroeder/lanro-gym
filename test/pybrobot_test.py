from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.robots import Panda


def test_panda_robot_state_obs():
    sim = PyBulletSimulation()
    panda1 = Panda(sim, full_state=False, fixed_gripper=True)
    panda2 = Panda(sim, full_state=False, fixed_gripper=False)
    panda3 = Panda(sim, full_state=True, fixed_gripper=True)
    panda4 = Panda(sim, full_state=True, fixed_gripper=False)

    assert panda1.get_obs().size == 6
    assert panda2.get_obs().size == 7
    assert panda3.get_obs().size == 19
    assert panda4.get_obs().size == 20

    assert panda1.action_space.shape == (7, )
    assert panda2.action_space.shape == (8, )
    assert panda3.action_space.shape == (7, )
    assert panda4.action_space.shape == (8, )

    assert panda1.get_ee_position().shape == (3, )
    assert panda1.get_ee_velocity().shape == (3, )

    assert panda2.get_ee_position().shape == (3, )
    assert panda2.get_ee_velocity().shape == (3, )

    assert panda3.get_ee_position().shape == (3, )
    assert panda3.get_ee_velocity().shape == (3, )

    assert panda4.get_ee_position().shape == (3, )
    assert panda4.get_ee_velocity().shape == (3, )

    assert panda1.get_current_pos().shape == (7, )
    assert panda2.get_current_pos().shape == (7, )
    assert panda3.get_current_pos().shape == (7, )
    assert panda4.get_current_pos().shape == (7, )

    assert panda1.get_fingers_width() == 0.0
    assert panda2.get_fingers_width() >= 0.0
    assert panda3.get_fingers_width() == 0.0
    assert panda4.get_fingers_width() >= 0.0

    assert panda1.get_obs().shape == (6, )
    assert panda2.get_obs().shape == (7, )
    assert panda3.get_obs().shape == (19, )
    assert panda4.get_obs().shape == (20, )
