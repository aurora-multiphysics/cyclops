import sys
import os

# Sort the paths out to run from this file
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(os.path.sep,parent_path, 'src')
sys.path.append(src_path)

from sim_reader import MeshReader, Unfolder
import numpy as np
import unittest
import meshio




class TestMeshReader(unittest.TestCase):
    def setUp(self):
        parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.__file_path = os.path.join(os.path.sep, parent_path, 'simulation', 'monoblock_out.e')
        self.__reader = MeshReader(self.__file_path)
                
    
    def test_read(self):
        pass


    def tearDown(self):
        pass



if __name__ == "__main__":
    unittest.main()