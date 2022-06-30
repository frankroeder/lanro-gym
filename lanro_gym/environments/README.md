# Environment

## Available robots

- `Panda`, the Franka Panda Emika robot

## Goal-conditioned tasks

- `Reach`
- `Push`
- `Slide`
- `PickAndPlace`
- `Stack{2..4}`

## Language-conditioned tasks

- `NLReach`
- `NLPush`
- `NLGrasp`
- `NLLift`

### Schema

The following schema is used to specify the environments for `gym.make()`:

`{Robot Name}{Task Name}{Number of Objects}[Mode][Observation Type][Hindsight Instructions]`.

Parameters in `{}` are required and `[]` are optional.

### Number of objects
- 2
- 3

### Mode
- ` ` (square with 3 colors)
- `Color` (square with 9 colors)
- `Shape` (3 shapes with 3 colors)
- `ColorShape` (3 shapes with 9 colors)

## Observation type
- (default): empty string to use world state (incl. kinematic information) as observation
- `PixelStatic` (static RGB image of the scene)
- `PixelEgo` (egocentric RGB image)

## Hindsight Instructions
- empty string
- `HI` to add hindsight intructions to the `info` dict of `gym.step()` when interaction with wrong object is detected
