from lanro.simulation import PyBulletSimulation
from lanro.robots import Panda


def test_panda_robot_state_obs():
    sim = PyBulletSimulation()
    panda1 = Panda(sim, full_state=False, fixed_gripper=True)
    assert panda1.get_obs().size == 6

    panda2 = Panda(sim, full_state=True, fixed_gripper=True)
    assert panda2.get_obs().size == 15
