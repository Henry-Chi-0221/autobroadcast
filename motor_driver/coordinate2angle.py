
import time
from matplotlib.pyplot import vlines
import numpy as np


class Coordinate2Angle():
    def __init__(self, left_angle: int = -45, right_angle: int = 45) -> None:
        """convert coordinates to angles

        Args:
            left_angle (int): camera takes the maximum angle to the left court
            left_angle (int): camera takes the maximum angle to the right court
        """

        self.left_ratio = -np.tan(np.deg2rad(left_angle))
        self.right_ratio = np.tan(np.deg2rad(right_angle))

        # print(self.left_ratio, self.right_ratio)

    def __call__(self, value: float) -> float:
        """give a value(-100 ~ 100) to get the angle the camera needs to rotation

        Args:
            value (int): -100 ~ 100

        Returns:
            float: the round(angle, 2) 
        """

        value /= 100
        if value > 0:
            value *= self.right_ratio
        elif value < 0:
            value *= self.left_ratio
        else:
            return 0
        angle = np.arctan(value) * 180 / np.pi
        return round(angle, 2)
