from typing import Any, List, Tuple, Optional, Union
import numpy as np
from enum import Enum


def scale_rgb(rgb_lst: List[float]) -> List[float]:
    return [_color / 255.0 for _color in rgb_lst]


class RGBCOLORS(Enum):
    """ RGBColors enum class with all colors defined as array of floats [0, 1]"""
    BLACK = scale_rgb([0, 0, 0])
    BLUE = scale_rgb([78.0, 121.0, 167.0])
    BROWN = scale_rgb([156.0, 117.0, 95.0])
    CYAN = scale_rgb([118.0, 183.0, 178.0])
    GRAY = scale_rgb([186.0, 176.0, 172.0])
    GREEN = scale_rgb([89.0, 169.0, 79.0])
    PINK = scale_rgb([255.0, 157.0, 167.0])
    ORANGE = scale_rgb([242.0, 142.0, 43.0])
    PURPLE = scale_rgb([176.0, 122.0, 161.0])
    RED = scale_rgb([255.0, 87.0, 89.0])
    WHITE = scale_rgb([255, 255, 255])
    YELLOW = scale_rgb([237.0, 201.0, 72.0])


class SHAPES(Enum):
    """ SHAPES enum class with all shapes with the corresponding object file id and words"""
    SQUARE = 0, ["box", "block", "square"],
    RECTANGLE = 1, ["rectangle", "oblong", "brick"],
    CYLINDER = 2, ["cylinder", "barrel", "tophat"],


def get_one_hot_list(total_items: int) -> np.ndarray:
    """Create an array with `total_items` one-hot vectors."""
    return np.eye(total_items)


def goal_distance(goal_a: np.ndarray, goal_b: np.ndarray) -> float:
    assert goal_a.shape == goal_b.shape, "mismatch of goal shapes"
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def post_process_camera_pixel(px, _height: int, _width: int) -> np.ndarray:
    rgb_array = np.array(px, dtype=np.uint8).reshape(_height, _width, 4)
    return rgb_array[:, :, :3]


def gripper_camera(bullet_client, projectionMatrix, pos, orn, imgsize: int = 84, mode: str = 'ego') -> np.ndarray:
    if mode == 'static':
        pos = [0, 0, 0]
        distance = 0.6
        yaw = 90
        pitch = -40
    elif mode == 'ego':
        distance = 0.5
        yaw = -45
        pitch = -30
    viewMatrix = bullet_client.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=pos,
                                                                 distance=distance,
                                                                 yaw=yaw,
                                                                 pitch=pitch,
                                                                 roll=0,
                                                                 upAxisIndex=2)
    (_, _, px, _, _) = bullet_client.getCameraImage(imgsize,
                                                    imgsize,
                                                    viewMatrix,
                                                    projectionMatrix,
                                                    flags=bullet_client.ER_NO_SEGMENTATION_MASK,
                                                    shadow=0,
                                                    renderer=bullet_client.ER_BULLET_HARDWARE_OPENGL)
    return post_process_camera_pixel(px, imgsize, imgsize)


def environment_camera(bullet_client, projectionMatrix, viewMatrix, width: int = 500, height: int = 500) -> np.ndarray:
    (_, _, px, _, _) = bullet_client.getCameraImage(width,
                                                    height,
                                                    viewMatrix,
                                                    projectionMatrix,
                                                    flags=bullet_client.ER_NO_SEGMENTATION_MASK,
                                                    renderer=bullet_client.ER_BULLET_HARDWARE_OPENGL)
    return post_process_camera_pixel(px, width, height)


class TaskObject:

    def __init__(self,
                 sim,
                 color: RGBCOLORS = RGBCOLORS.RED,
                 onehot_color: np.ndarray = np.array([]),
                 shape: Optional[SHAPES] = SHAPES.SQUARE,
                 onehot_shape: Optional[np.ndarray] = np.array([]),
                 obj_mass: int = 2,
                 object_size: float = 0.04):
        self.sim = sim
        self.secondary = shape
        self.secondary_onehot = onehot_shape

        self.color = color
        self.onehot_color = onehot_color

        self.primary = color
        self.primary_onehot = onehot_color

        self.shape = shape
        self.onehot_shape = onehot_shape

        self.object_size = object_size
        self.obj_mass = obj_mass

    def load(self, object_body_key):
        if self.shape:
            shape_id = self.shape.value[0]
            if shape_id == SHAPES.SQUARE.value[0]:
                self.sim.create_box(
                    body_name=object_body_key,
                    half_extents=[
                        self.object_size / 2,
                        self.object_size / 2,
                        self.object_size / 2,
                    ],
                    mass=self.obj_mass,
                    position=[0.0, 0.0, self.object_size / 2],
                    rgba_color=self.color.value + [1],
                )
            elif shape_id == SHAPES.RECTANGLE.value[0]:
                self.sim.create_box(
                    body_name=object_body_key,
                    half_extents=[
                        self.object_size / 2 * 2,
                        self.object_size / 2 * 0.75,
                        self.object_size / 2 * 0.75,
                    ],
                    mass=self.obj_mass,
                    position=[0.0, 0.0, self.object_size / 2],
                    rgba_color=self.color.value + [1],
                )
            elif shape_id == SHAPES.CYLINDER.value[0]:
                self.sim.create_cylinder(
                    body_name=object_body_key,
                    radius=self.object_size * 0.5,
                    height=self.object_size * 0.75,
                    mass=self.obj_mass * 3,
                    position=[0.0, 0.0, self.object_size / 2],
                    rgba_color=self.color.value + [1],
                    lateral_friction=1.0,
                    spinning_friction=0.005,
                )
        else:
            self.sim.create_box(
                body_name=object_body_key,
                half_extents=[
                    self.object_size / 2,
                    self.object_size / 2,
                    self.object_size / 2,
                ],
                mass=self.obj_mass,
                position=[0.0, 0.0, self.object_size / 2],
                rgba_color=self.color.value + [1],
            )

    def get_onehot(self):
        return np.concatenate([self.primary_onehot, self.secondary_onehot])

    def get_properties(self) -> Tuple:
        return self.primary, self.secondary

    def get_color(self) -> RGBCOLORS:
        return self.color

    def get_shape(self) -> Union[SHAPES, Any]:
        return self.shape


class TaskObjectList:

    def __init__(self, sim, color_mode: bool = False, shape_mode: bool = False):

        # default colors
        task_colors = [RGBCOLORS.RED, RGBCOLORS.GREEN, RGBCOLORS.BLUE]
        if color_mode:
            # extend range of colors
            task_colors.extend([
                RGBCOLORS.YELLOW,
                RGBCOLORS.PURPLE,
                RGBCOLORS.ORANGE,
                RGBCOLORS.PINK,
                RGBCOLORS.CYAN,
                RGBCOLORS.BROWN,
            ])
        onehot_colors = get_one_hot_list(len(task_colors))

        # shape mode combinations
        if shape_mode:
            onehot_shapes = get_one_hot_list(len(SHAPES))
            self.objects = [
                TaskObject(sim,
                           color=color,
                           onehot_color=onehot_colors[ci],
                           shape=shape,
                           onehot_shape=onehot_shapes[si]) for ci, color in enumerate(task_colors)
                for si, shape in enumerate(SHAPES)
            ]
        else:
            # default setup with one shape
            self.objects = [
                TaskObject(sim, color=color, onehot_color=onehot_colors[ci]) for ci, color in enumerate(task_colors)
            ]

    def get_obj_properties(self):
        return [obj.get_properties() for obj in self.objects]

    def __getitem__(self, index):
        return self.objects[index]

    def __len__(self):
        return len(self.objects)
