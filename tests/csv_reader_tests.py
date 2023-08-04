from model_management import CSVReader
import pandas as pd
import numpy as np
import unittest
import os




class TestCSVReader(unittest.TestCase):
    def setUp(self):
        # We produce the temperature field below
        # 0.2   1.2     2.2
        # 0.1           2.1
        # 0     1.0     2.0
        df = pd.DataFrame({
            'X': [0, 0, 0, 1, 1, 2, 2, 2],
            'Y': [0, 1, 2, 0, 2, 0, 1, 2],
            'T': [0, 0.1, 0.2, 1.0, 1.2, 2.0, 2.1, 2.2]
        })

        script_path = os.path.realpath(__file__)
        parent_path = os.path.dirname(os.path.dirname(script_path))
        file_path = os.path.join(os.path.sep,parent_path,"src","test_temperature.csv")
        print(file_path)
        df.to_csv(file_path, index=False)
        self.__csv_reader = CSVReader('test_temperature.csv')

    
    def test_find_nearest_pos(self):
        nearest_pos = self.__csv_reader.find_nearest_pos(np.array([0.0, 2.0]))
        test_answer = (0.0, 2.0)
        self.assertEqual(nearest_pos[0], test_answer[0])
        self.assertEqual(nearest_pos[1], test_answer[1])

        nearest_pos = self.__csv_reader.find_nearest_pos(np.array([0.01, 2.01]))
        test_answer = (0.0, 2.0)
        self.assertEqual(nearest_pos[0], test_answer[0])
        self.assertEqual(nearest_pos[1], test_answer[1])

        nearest_pos = self.__csv_reader.find_nearest_pos(np.array([1.001, 1.01]))
        test_answer = (1.0, 2.0)
        self.assertEqual(nearest_pos[0], test_answer[0])
        self.assertEqual(nearest_pos[1], test_answer[1])

    
    def test_get_temp(self):
        csv_temp = self.__csv_reader.get_temp(self.__csv_reader.find_nearest_pos(np.array([0.0, 2.0])))
        self.assertEqual(csv_temp, 0.2)

        csv_temp = self.__csv_reader.get_temp(self.__csv_reader.find_nearest_pos(np.array([0.01, 2.1])))
        self.assertEqual(csv_temp, 0.2)

        csv_temp = self.__csv_reader.get_temp(self.__csv_reader.find_nearest_pos(np.array([1.0001, 1.01])))
        self.assertEqual(csv_temp, 1.2)


    def test_get_loss(self):
        pass



    def tearDown(self):
        script_path = os.path.realpath(__file__)
        parent_path = os.path.dirname(os.path.dirname(script_path))
        file_path = os.path.join(os.path.sep,parent_path,"src","test_temperature.csv")
        os.remove(file_path)



if __name__ == "__main__":
    unittest.main()