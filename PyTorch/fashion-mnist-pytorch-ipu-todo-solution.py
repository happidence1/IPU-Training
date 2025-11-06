# Modified from Graphcore tutorials https://github.com/graphcore/tutorials.git

# Import the packages

import torch
import torchvision
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PopTorch is an extended and separate package from PyTorch.
# It is available in Graphcore's Poplar SDK

# Todo: Step 1 - import the PopTorch package.
# Write your code below
import poptorch

# Note: Under the hood, PopTorch uses Graphcore's high-performance machine learning framework PopART. 
#       It is therefore necessary to enable PopART and Poplar in your environment.


# Load the data

# We will use the torchvision built-in Fashion-MNIST dataset
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = torchvision.datasets.FashionMNIST(
    "~/.torch/datasets", transform=transform, download=True, train=True
)

test_dataset = torchvision.datasets.FashionMNIST(
    "~/.torch/datasets", transform=transform, download=True, train=False
)

classes = (
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

# Build the model
# We will build a simple CNN model for classification. 

class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 12, 5)
        self.norm = nn.GroupNorm(3, 12)
        self.fc1 = nn.Linear(972, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def forward(self, x, labels=None):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.norm(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        # We include the loss computation in the forward function to make sure it is performed on IPU not CPU
        # Todo: Step 2 - add the loss computation
        if self.training:
            # Write your code below
            return x, self.loss(x, labels)
            
        return x


model = ClassificationModel()
model.train()


# Prepare training for IPUs

# The compilation and execution on the IPU can be controlled using poptorch.Options. 
# These options are used by PopTorch's wrappers such as poptorch.DataLoader and poptorch.trainingModel

# Todo: Step 3 - set up the poptorch.Options
# Write your code below
opts = poptorch.Options()

# PopTorch DataLoader
# PopTorch offers an extension of torch.utils.data.DataLoader with its poptorch.DataLoader class, 
# specialised for the way the underlying PopART framework handles batching of data.

# Todo: Step 4 - create train_dataloader with poptorch.Options
# Write your code below
train_dataloader = poptorch.DataLoader(
    opts, train_dataset, batch_size=16, shuffle=True, num_workers=20
)

# Train the model

# We will need another component in order to train our model: an optimiser. 
# Its role is to apply the computed gradients to the model's weights to optimize (usually, minimize) the loss function using a specific algorithm. 
# PopTorch currently provides classes which inherit from multiple native PyTorch optimisation functions: SGD, Adam, AdamW, LAMB and RMSprop.
# We will use SGD as it's a very popular algorithm and is appropriate for this classification task.

# Todo: Step 5 - set up the optimizer with poptorch.optim.xxx
# Write your code below
optimizer = poptorch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Todo: Step 6 - Use the poptorch.trainingModel wrapper for the poptorch_model. It takes model, opts and optimizer.
# Write your code below
poptorch_model = poptorch.trainingModel(model, options=opts, optimizer=optimizer)

# Traing loop
epochs = 30
for epoch in tqdm(range(epochs), desc="epochs"):
    total_loss = 0.0
    for data, labels in tqdm(train_dataloader, desc="batches", leave=False):
        output, loss = poptorch_model(data, labels)
        total_loss += loss

# Todo: Step 7 -  detach the model from the device to use the same IPU for training and inference
# Write your code below
poptorch_model.detachFromDevice()

# Save the trained model
torch.save(model.state_dict(), "classifier.pth")

# Evaluate the model
# The steps taken below to define the model for evaluation essentially allow it to run in inference mode. 
model = model.eval()

# Todo: Step 8 - Use the poptorch.inferenceModel wrapper for the poptorch_model_inf. It takes model, and opts.
# Write your code below
poptorch_model_inf = poptorch.inferenceModel(model, options=opts)

# Todo: Step 9 - create test_dataloader with poptorch.Options
# Write your code below
test_dataloader = poptorch.DataLoader(opts, test_dataset, batch_size=32, num_workers=10)

predictions, labels = [], []

for data, label in test_dataloader:
    predictions += poptorch_model_inf(data).data.max(dim=1).indices
    labels += label

# Todo: Step 10 -  detach the model from the device.
# Write your code below
poptorch_model_inf.detachFromDevice()

print(f"Eval accuracy: {100 * accuracy_score(labels, predictions):.2f}%")