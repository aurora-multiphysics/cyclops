import pickle
import os


class PickleManager():
    """
    Reads and writes to pickle files.
    """
    def correct_path(self, folder :str, file_name :str) -> os.PathLike:
        """
        Finds the absolute path for a certain file.

        Args:
            folder (str): path relative to the directory (folder name).
            file_name (str): name of file.

        Returns:
            os.PathLike: absolute path.
        """
        dir_path = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(os.path.sep, dir_path, folder, file_name)


    def save_file(self, folder :str, file_name :str, save_object :object) -> None:
        """
        Saves an object into a pickle file.

        Args:
            folder (str): folder to save the object in.
            file_name (str): name of the object file.
            save_object (object): object to save.
        """
        full_path = self.correct_path(folder, file_name)
        save_file = open(full_path, 'wb')
        pickle.dump(save_object, save_file)
        save_file.close()


    def read_file(self, folder :str, file_name :str) -> object:
        """
        Reads an object file and returns the object.

        Args:
            folder (str): folder where the object is.
            file_name (str): name of object file.

        Returns:
            object: object.
        """
        full_path = self.correct_path(folder, file_name)
        read_file = open(full_path, 'rb')
        read_object = pickle.load(read_file)
        read_file.close()
        return read_object