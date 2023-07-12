from chigger.exodus.ExodusSourceLineSampler import ExodusSourceLineSampler
from chigger.exodus.ExodusSource import ExodusSource
from chigger.exodus.ExodusReader import ExodusReader




if __name__ == "__main__":
    reader = ExodusReader("file.e")
    file_in = ExodusSource(reader)
    sampler = ExodusSourceLineSampler(file_in)