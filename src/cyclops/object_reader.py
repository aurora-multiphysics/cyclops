"""
PickleManager class for cyclops.

Handles serialisation of objects.

(c) Copyright UKAEA 2023.
"""
import pickle
import pathlib


class PickleManager:
    """Read and write .pickle files."""

    def save_file(self, file_path: str, object_to_save: object) -> None:
        """Serialise an object instance to a .pickle file.

        Args:
            file_path (str): absolute or relative path to the file to write.
            object_to_save (object): object instance to write to file.
        """
        file_path = pathlib.Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(object_to_save, file)

    def read_file(self, file_path: str) -> object:
        """Read an object instance from a .pickle file and return it.

        Args:
            file_path (str): absolute or relative path to the file to read.

        Returns:
            object_to_return (object): the object instance loaded from file.
        """
        file_path = pathlib.Path(file_path)
        with open(file_path, "rb") as file:
            object_to_return = pickle.load(file)
        return object_to_return
