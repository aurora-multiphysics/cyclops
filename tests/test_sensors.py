import sys
import os

# Sort the paths out to run from this file
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(os.path.sep,parent_path, 'src')
sys.path.append(src_path)

from sensors2 import PointSensor1D, PointSensor2D, RoundSensor1D, MultiSensor1D, MultiSensor2D, Thermocouple1D, Thermocouple2D
import numpy as np
import unittest




class Test(unittest.TestCase):
    def test_point(self):
        pass





if __name__ == "__main__":
    unittest.main()