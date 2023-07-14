from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
import matplotlib.pyplot as plt
import numpy as np



def f(x):
    return np.exp(x)



class NoiselessGPModel():
    def __init__(self, x_train, y_train):
        kernel = RationalQuadratic(length_scale=1)
        self.__gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        self.__gp.fit(x_train, y_train)
        self.__x_train = x_train
        self.__y_train = y_train

    
    def predict(self, x_test, std_out=True):
        return self.__gp.predict(x_test, return_std=std_out)

    
    def plot(self, x_test = np.linspace(-5, 5, 100).reshape(100, 1)):
        test_y, std_y = self.__gp.predict(x_test, return_std=True)

        plt.plot(self.__x_train, self.__y_train, 'k*')
        plt.plot(x_test, test_y, 'b')
        plt.fill_between(
            x_test.ravel(),
            test_y - 1.96 * std_y,
            test_y + 1.96 * std_y,
            alpha=0.95,
        )
        plt.legend(['Observed Data', 'Mean', r'95% confidence interval'])
        #plt.plot(x_test, f(x_test), 'k--')
        plt.show()
        plt.close()



if __name__== "__main__":
    #x_train = np.linspace(-5, 5).reshape(-1, 1)
    #y_train = f(x_train)
    x_train = np.array([0.011499999999999996, -0.011500000000000002, 0.009583333333333327, 0.010541666666666661, 0.007666666666666663, 0.008624999999999996, 0.0057499999999999956, 0.006708333333333329, 0.0038333333333333297, 0.004791666666666663, 0.001916666666666662, 0.0028749999999999956, -3.469446951953614e-18, 0.0009583333333333293, -0.0019166666666666707, -0.0009583333333333371, -0.0038333333333333353, -0.002875000000000003, -0.005750000000000002, -0.004791666666666668, -0.007666666666666669, -0.006708333333333335, -0.009583333333333333, -0.008625, -0.010541666666666668, 0.0125, 0.011999999999999999, 0.0135, 0.013000000000000001, 0.014499999999999999, 0.013999999999999999, 0.0155, 0.015, 0.0165, 0.016, 0.0175, 0.017, 0.0185, 0.018000000000000002, 0.0195, 0.019])
    y_train = np.array([617.7810142261592, 170.6022114838236, 521.7502792402029, 568.8989262268101, 434.0389864587067, 476.6345507104152, 357.9193076223358, 394.309551443684, 296.316270176559, 325.16970344591056, 250.4265916926481, 271.4404941868866, 218.85814886792284, 233.03399846017842, 198.3958836666498, 207.4599397658493, 185.64295962080178, 191.234545552002, 177.9572861156409, 181.28869236112325, 173.5446931329063, 175.4183977941438, 171.2913781739981, 172.19144884991806, 170.76938841367416, 670.2457401119626, 643.8438526191087, 723.9169566473661, 696.9475635514357, 778.5447183940522, 751.1245377159483, 833.9347121516801, 806.1546527061431, 889.9389747620435, 861.8676598387348, 946.449324461157, 918.1362163868741, 1003.3922629105379, 974.8700538530743, 1060.7250688497088, 1032.0114913020122])
    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)

    gp = NoiselessGPModel(x_train, y_train)
    gp.plot(x_test = np.linspace(-0.012, 0.02, 100).reshape(100, 1))
    #gp.plot()