import pickle
import os


class PickleManager():
    def correct_path(self, folder, file_name):
        dir_path = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(os.path.sep, dir_path, folder, file_name)


    def save_file(self, folder, file_name, save_object):
        full_path = self.correct_path(folder, file_name)
        save_file = open(full_path, 'wb')
        pickle.dump(save_object, save_file)
        save_file.close()


    def read_file(self, folder, file_name):
        full_path = self.correct_path(folder, file_name)
        read_file = open(full_path, 'rb')
        read_object = pickle.load(read_file)
        read_file.close()
        return read_object