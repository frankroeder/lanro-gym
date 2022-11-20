# LANRO (Language Robotics)
<div>
<img src="./docs/panda_nlpush.gif" width="100%" height="auto">
</div>

__LANRO__ is a platform to study language-conditioned reinforcement learning.
It is part of the following publications that introduced the following features:
1. a synthetic caretaker providing instructions in hindsight [**_Grounding Hindsight Instructions in Multi-Goal Reinforcement Learning for Robotics_**](https://arxiv.org/abs/2204.04308) ([ICDL 2022](https://icdl2022.qmul.ac.uk/), see `icdl2022` branch for old version)
2. a setup for conversational repair via action corrections [**_Language-Conditioned Reinforcement Learning to Solve Misunderstandings with Action Corrections_**](https://openreview.net/forum?id=lWd0qiv9E-) ([NeurIPS 2022 Workshop LaReL](https://larel-workshop.github.io/))

## Installation
### Pip module

```bash
pip install lanro-gym
```

### From source

```bash
git clone https://github.com/frankroeder/lanro-gym.git
cd lanro-gym/  && pip install -e .
```
or
```bash
# via https
pip install git+https://github.com/frankroeder/lanro-gym.git
# or ssh
pip install git+ssh://git@github.com/frankroeder/lanro-gym.git
```

## Example

```python
import gymnasium as gym
import lanro_gym

env = gym.make('PandaStack2-v0', render=True)

obs, info = env.reset()
terminated = False
while not terminated:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

env.close()
```

## Environments

[Click here for the environments README](./lanro_gym/environments/README.md)

## Keyboard and mouse control
It is also possible to manipulate the robot with sliders
```bash
python main.py -i --env PandaNLReach2-v0
```
or your keyboard
```bash
python main.py -i --keyboard --env PandaNLReach2-v0
```

## Developers

### Running tests

We use [pytest](https://realpython.com/pytest-python-testing/).
```bash
PYTHONPATH=$PWD pytest test/
```

Measure the FPS of your system:
```bash
PYTHONPATH=$PWD python examples/fps.py
```

## Acknowledgements

This work uses code and got inspired by following open-source projects:

#### pybullet

Homepage [https://pybullet.org/](https://pybullet.org/)  
Source: [https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet)  
License: [Zlib](http://opensource.org/licenses/Zlib)  

#### panda-gym

Source: [https://github.com/qgallouedec/panda-gym](https://github.com/qgallouedec/panda-gym)  
License: [MIT](https://github.com/qgallouedec/panda-gym/blob/master/LICENSE)  
Changes: The code structure of `lanro-gym` contains copies and extensively modified parts of `panda-gym`.

## Citations

**Grounding Hindsight Instructions in Multi-Goal Reinforcement Learning for Robotics**
```bibtex
@inproceedings{Roder_GroundingHindsight_2022,
  title = {Grounding {{Hindsight Instructions}} in {{Multi-Goal Reinforcement Learning}} for {{Robotics}}},
  booktitle = {International {{Conference}} on {{Development}} and {{Learning}}},
  author = {R{\"o}der, Frank and Eppe, Manfred and Wermter, Stefan},
  year = {2022},
  pages = {170--177},
  publisher = {{IEEE}},
  isbn = {978-1-66541-310-7},
}
```

**pybullet**
```bibtex
@MISC{coumans2021,
  author = {Erwin Coumans and Yunfei Bai},
  title = {PyBullet, a Python module for physics simulation for games, robotics and machine learning},
  howpublished = {\url{http://pybullet.org}},
  year = {2016--2021}
}
```

**panda-gym**
```bibtex
@article{gallouedec2021pandagym,
  title   = {{panda-gym: Open-Source Goal-Conditioned Environments for Robotic Learning}},
  author  = {Gallou{\'e}dec, Quentin and Cazin, Nicolas and Dellandr{\'e}a, Emmanuel and Chen, Liming},
  year    = 2021,
  journal = {4th Robot Learning Workshop: Self-Supervised and Lifelong Learning at NeurIPS},
}
```
