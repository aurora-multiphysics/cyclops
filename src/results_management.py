import os



class ResultsManager():
    def __init__(self, file_name):
        parent_path = os.path.dirname(os.path.dirname(__file__))
        self.__setups_path = os.path.join(os.path.sep,parent_path, 'results', file_name)

        # Read the file
        self.__IDs = []
        self.__models = []
        self.__sensor_layouts = []

        f = open(self.__setups_path, "r")
        for i, line in enumerate(f):
            line = line.rstrip()
            if line != '':
                if line[0:4] == 'ID: ':
                    self.__IDs.append(line[4:])
                elif line[0:7] == 'Model: ':
                    self.__models.append(line[7:])
                elif line[0] == '=':
                    self.__sensor_layouts.append([])
                elif line[0] == '[':
                    sensor_sequence = line[1:-1].split(', ')
                    for i, e in enumerate(sensor_sequence):
                        sensor_sequence[i] = float(e)    
                    self.__sensor_layouts[-1].append(sensor_sequence)
        f.close()


    def write_file(self, write_model, write_layouts):
        # Update the best result & setup for a certain number of sensors
        self.__IDs.append(str(int(self.__IDs[-1])+1))
        self.__models.append(write_model)
        self.__sensor_layouts.append(write_layouts)


    def read_file(self, read_ID):
        # Return the best result & setup for a certain number of sensors
        index = self.__IDs.index(read_ID)
        return self.__models[index], self.__sensor_layouts[index]


    def save_updates(self):
        # Save any updates into the file by rewriting to it
        f = open(self.__setups_path, "w")
        for i, id in enumerate(self.__IDs):
            f.writelines('ID: '+id+'\n')
            f.writelines('Model: '+self.__models[i]+'\n')
            f.writelines('=================='+'\n')
            for layout in self.__sensor_layouts[i]:
                f.writelines(str(layout)+'\n')
            f.writelines('\n')
        f.close()


    def get_IDs(self):
        return self.__IDs

