from matplotlib import pyplot as plt
import gpytorch
import torch



class PlaneModel():
    def __init__(self, pos1, pos2, pos3):
        # A plane is defined by Ax + By + CT = D where A, B, C are the components of the normals and D is a constant
        # Hence to find the T at a specific position in the plane all we need is T = (D - Ax - By)/C
        normal = torch.cross(pos2 - pos1, pos3 - pos1)
        self.__A = normal[0]
        self.__B = normal[1]
        self.__C = normal[2]
        self.__D = torch.dot(normal, pos1)
    

    def get_T(self, pos_xy):
        # We return the temperature at a specific x and y position
        temperature_tensor = (self.__D - self.__A * pos_xy[:, 0] - self.__B * pos_xy[:, 1])/self.__C
        return torch.reshape(temperature_tensor, (-1, 1))






class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-10))):
        super().__init__(x_train, y_train, likelihood=likelihood)
        self.__x_train = x_train
        self.__y_train = y_train

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = likelihood

        self.likelihood.noise = 1e-3
        all_params = set(self.parameters())
        final_params = list(all_params - {self.likelihood.raw_noise})
        self.optimiser = torch.optim.Adam(final_params, lr=0.1)

        
    
    def forward(self, x):
        # Feed forward the input x
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    def learn(self, repetitions=50):
        # Train the model
        print("\nTraining...")
        self.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        for i in range(repetitions):
            self.optimiser.zero_grad()
            output = self(self.__x_train)
            loss = -mll(output, self.__y_train)
            loss.backward()
            self.optimiser.step()
        print("\nResults")
        print("Loss: "+str(loss.item()))
        print("Lengthscale: "+str(self.covar_module.base_kernel.lengthscale.item()))
        print("Noise: "+str(self.likelihood.noise.item())+"\n")


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

            plt.plot(self.__x_train.numpy(), self.__y_train, 'k*')
            plt.plot(test_x.numpy(), test_y.mean.numpy(), 'b')
            plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            plt.legend(['Observed Data', 'Mean', 'Confidence'])
            plt.show()
            plt.close()





if __name__ == "__main__":
    plane = PlaneModel(torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4]), torch.tensor([2, 4, 5]))
    print(plane.get_T(torch.tensor([[1, 2], [2, 3], [2, 4]])))
