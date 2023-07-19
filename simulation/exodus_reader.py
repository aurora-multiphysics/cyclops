from matplotlib import pyplot as plt
import chigger
import torch
import os








if __name__ == "__main__":
    relative_path = 'monoblock_out11.e'
    absolute_path = os.path.dirname(__file__)
    full_path = os.path.join(absolute_path, relative_path)
    reader = chigger.exodus.ExodusReader(full_path)
    reader.update()

    result = chigger.exodus.ExodusResult(reader, variable='temperature', viewport=[0,0,0.5,1])
    cbar = chigger.exodus.ExodusColorBar(result)
    result.update()

    sample = chigger.exodus.ExodusResultLineSampler(result, resolution=200, point1 = (0, -0.015, 0), point2 = (0.0, 0.02, 0.0))
    sample.update()

    x = sample[0].getDistance()
    y = sample[0].getSample('temperature')

    line = chigger.graphs.Line(x, y, width=4, label='probe')
    graph = chigger.graphs.Graph(line, yaxis={'lim':[0,1600]}, xaxis={'lim':[0,0.04]}, viewport=[0.5,0,1,1])

    plt.plot(x, y)
    plt.show()
    plt.close()