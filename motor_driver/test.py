import subprocess as sp
from loguru import logger
from serial import Serial
import time

import time
import numpy as np
from random import randint


class Coordinate2Angle():
    def __init__(self, left_angle: int = -45, right_angle: int = 45) -> None:
        """ convert coordinates to angles
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


class ArduinoControl():
    def __init__(self):
        self.connect = self.connect_arduino()
        time.sleep(10)
        for i in range(3):
            print(self.read())
        logger.info("Arduino connected")

    def connect_arduino(self, baudrate=115200):
        # check device is connected to computer
        command = "ls /dev | grep ttyACM*"
        arduino_device = sp.run(command,
                                shell=True,
                                encoding="utf - 8",
                                stdout=sp.PIPE,
                                stderr=sp.PIPE)
        arduino_device = arduino_device.stdout.split()
        if len(arduino_device) == 0:
            logger.error("Could not find Arduino device")
            raise ConnectionError("Could not find Arduino device")
        logger.info("Arduino port set to /dev/" + arduino_device[0])
        connect = Serial(port='/dev/' + arduino_device[0],
                         baudrate=baudrate,
                         timeout=.1)
        return connect

    def int16_to_bytes(self, i: int):
        return i >> 8, i % 256

    def write(self, data):
        # data = bytearray([*self.int16_to_bytes(int(data))])
        data = (str(data) + "\n").encode('ascii')
        self.connect.write(data)

    def read(self):
        return self.connect.readline()

if __name__ == "__main__":
    coor2angle = Coordinate2Angle()
    arduino = ArduinoControl()
    upper = True
    a = 0
    while(1):
        
        if upper and a<100:
            a += 1
        if a == 100:
            upper = False
        if not upper and a > -100:
            a -= 1
        if a == -100:
            upper = True
        #a = input("Enter a value between -100 & 100 \n")
        
        deg = coor2angle(int(a))
        print(deg)
        arduino.write(deg)
        r = arduino.read()