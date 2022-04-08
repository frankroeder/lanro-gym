# LANRO (Language Robotics)
<div>
<img src="./docs/panda_nlpush.gif" width="100%" height="auto">
</div>

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
import gym
import lanro

env = gym.make('PandaStack2-v0', render=True)

obs = env.reset()
done = False
while not done:
    obs, reward, done, info = env.step(env.action_space.sample())

env.close()
```

## Environments

[Click here for the environments README](./lanro/environments/README.md)

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
