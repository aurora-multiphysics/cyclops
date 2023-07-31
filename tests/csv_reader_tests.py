from src.csv_reader import CSVReader
import pandas as pd
import unittest
import os




class TestCSVReader(unittest.TestCase):
    def setUp(self):
        # We produce the temperature field below
        # 2.0   2.1     2.2
        # 1.0           1.2
        # 0     0.1     0.2
        df = pd.dataframe({
            'X': [0, 0, 0, 1, 1, 2, 2, 2],
            'Y': [0, 1, 2, 0, 2, 0, 1, 2],
            'T': [0, 0.1, 0.2, 1.0, 1.2, 2.0, 2.1, 2.2]
        })

        absolute_path = os.path.dirname(__file__)
        full_path = os.path.join(absolute_path, 'test_temperature.csv')
        df.to_csv(full_path, index=False)

        self.__csv_reader = CSVReader('test_temperature.csv')

    
    def test_find_nearest_pos(self):
        


    def tearDown(self):
        absolute_path = os.path.dirname(__file__)
        full_path = os.path.join(absolute_path, 'test_temperature.csv')
        os.remove(full_path)