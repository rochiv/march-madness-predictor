import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

###########################################################################################################
# This is an example of using a pytorch neural network to classify images from the FashionMNIST Dataset
###########################################################################################################

# Training Data (contains ground truth)
training_data = datasets.FashionMNIST(
    root="FashionMNIST_data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Testing data (no ground truth)
test_data = datasets.FashionMNIST(
    root="FashionMNIST_data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# batch_size is how many data points are used to train at one time
batch_size = 64
# making DataLoader object with our Training Data
train_dataloader = DataLoader(training_data, batch_size=batch_size)
# making DataLoader object with our Testing Data
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Describing the Training and Testing DataLoaders
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu or gpu device for training. (Cuda = NVIDIA GPU)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Define model class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Define model
my_model = NeuralNetwork().to(device)
print(my_model)
# Loss function for model (we use Cross Entropy)
my_loss_fn = nn.CrossEntropyLoss()
# Optimizer for model (we use Stochastic Gradient Descent)
my_optimizer = torch.optim.SGD(my_model.parameters(), lr=1e-3)

# train function for our model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# test function for our model
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# calling our functions to train model
# epoch is 'training iterations'
epochs = 10
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, my_model, my_loss_fn, my_optimizer)
    test(test_dataloader, my_model, my_loss_fn)
print("Done!")

# save model
torch.save(my_model.state_dict(), "my_model.pth")
print("Saved PyTorch Model State to my_model.pth")