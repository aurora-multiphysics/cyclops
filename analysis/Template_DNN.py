from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from torch import nn
import pandas as pd
import torch


#Define important variables.
batch_size = 64
epochs = 10
loss_fn = nn.CrossEntropyLoss()
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")




#Define custom dataset.
class FashionSet(Dataset):
    def __init__(self, images, labels):
        self.__length = len(labels)
        self.__data = images
        self.__targets = labels
        

    def __len__(self):
        return self.__length


    def __getitem__(self, i):
        return self.__data[i], self.__targets[i]
    



# Define model.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)




#Define data collection.
def gather_data(path_to_csv):
    train_df = pd.read_csv(path_to_csv)
    images = train_df.drop("label", axis=1).values
    labels = train_df['label'].values

    images = torch.tensor(images, dtype=torch.float32)
    images /= 255.0
    labels = torch.tensor(labels, dtype=torch.long)

    images = images.view(len(images),1, 28, 28)
    labels = labels.view(len(labels))
    return train_test_split(images, labels, train_size=0.7, shuffle=True)



#Define training loop.
def train(dataloader, model, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")




#Define testing loop.
def test(dataloader, model):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")




def evaluate(model, dataloader):
    #Note that batch size for dataloader must be 1
    model.eval()
    with torch.no_grad():
        figure = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3
        for batch, (x, y) in enumerate(dataloader, 1):
            pred = model(x)
            predicted_category = int(pred[0].argmax(0))
            figure.add_subplot(rows, cols, batch)
            plt.title(labels_map[predicted_category])
            plt.axis("off")
            plt.imshow(x.squeeze(), cmap="gray")
        plt.show()
        plt.close()




if __name__ == "__main__":
    #Setup data
    x_train, x_test, y_train, y_test = gather_data("/home/domharrington/Documents/Practice/fashion/fashion-mnist_train.csv")
    train_dataloader = DataLoader(
        FashionSet(x_train, y_train), 
        batch_size=batch_size
    )
    test_dataloader = DataLoader(
        FashionSet(x_test, y_test), 
        batch_size=batch_size
    )
    print("\nData setup complete!")


    for x, y in test_dataloader:
        print(f"Shape of x [N, C, H, W]: {x.shape} {x.dtype}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break


    #Setup model
    model = NeuralNetwork().to(device)    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


    #Train model
    for n in range(epochs):
        print(f"Epoch {n+1}\n-------------------------------")
        train(train_dataloader, model, optimizer)
        test(test_dataloader, model)
    print("Done!")


    #Save model
    torch.save(model.state_dict(), "FashionModel3.pth")
    print("Saved PyTorch Model State to FashionModel3.pth")


    #Evaluate model
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("FashionModel3.pth"))
    eval_dataloader = DataLoader(
        FashionSet(x_test[:9], y_test[:9]),
        batch_size=1
    )
    evaluate(model, eval_dataloader)
    print("Evaluated!")
