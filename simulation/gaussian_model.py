from matplotlib import pyplot as plt
from tqdm import tqdm
import gpytorch
import torch




def f(x):
    return torch.sin(x)



class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        super().__init__(train_x, train_y, likelihood=likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.1)
        self.likelihood = likelihood
        self.__train_x = train_x
        self.__train_y = train_y
    

    def forward(self, x):
        # Feed forward the input x
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    def learn(self, repetitions=500):
        # Train the model
        print("\nTraining...")
        self.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        for i in tqdm(range(repetitions)):
            self.optimiser.zero_grad()
            output = self(self.__train_x)
            loss = -mll(output, self.__train_y)
            loss.backward()
            self.optimiser.step()


    def predict(self, test_x = torch.linspace(-5, 5, 100)):
        # Use the model to predict values
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self.likelihood(self(test_x))


    def plot(self, test_x = torch.linspace(-5, 5, 100)):
        # Plot the results of the model
        with torch.no_grad():
            test_y = self.predict(test_x)
            lower, upper = test_y.confidence_region()

            plt.plot(self.__train_x.numpy(), self.__train_y, 'k*')
            plt.plot(test_x.numpy(), test_y.mean.numpy(), 'b')
            plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            plt.legend(['Observed Data', 'Mean', 'Confidence'])
            plt.show()
            plt.close()





if __name__ == "__main__":
    train_x = torch.FloatTensor(10).uniform_(-5, 5)
    train_y = f(train_x)

    model = ExactGPModel(train_x, train_y)
    model.learn()
    model.plot()