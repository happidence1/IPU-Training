# IPU-Training

TAMU Graphcore IPU training

Credit to Graphcore tutorials https://github.com/graphcore/examples/tree/master/tutorials

## Code Snippets from Presentation

### Section II. Demo on ACES


#### Training Materials

- From the ACES login node, ssh into the poplar2 (BOW Pod16) IPU system

    ```
    ssh poplar2
    ```
- Change to your directory:
    ```
    cd /localdata/$USER && mkdir ipu_labs && cd ipu_labs
    ```
- Copy the example materials to your directory:
    ```
    git clone https://github.com/graphcore/examples.git
    ```
- Copy the hands-on exercise materials to your directory:
    ```
    git clone https://github.com/happidence1/IPU-Training.git
    ```

#### Poplar SDK setup

```
source /opt/gc/poplar/poplar_sdk-ubuntu_20_04-3.3.0+1403-208993bbb7/poplar-ubuntu_20_04-3.3.0+7857-b67b751185/enable.sh

source /opt/gc/poplar/poplar_sdk-ubuntu_20_04-3.3.0+1403-208993bbb7/popart-ubuntu_20_04-3.3.0+7857-b67b751185/enable.sh

mkdir -p /localdata/$USER/tmp
export TF_POPLAR_FLAGS=--executable_cache_path=/localdata/$USER/tmp
export POPTORCH_CACHE_DIR=/localdata/$USER/tmp
```

#### TF Virtual Environment Setup

```
virtualenv -p python3 venv_tf2

source venv_tf2/bin/activate

python -m pip install /opt/gc/poplar/poplar_sdk-ubuntu_20_04-3.3.0+1403-208993bbb7/tensorflow-2.6.3+gc3.3.0+251582+08d96978c7f+intel_skylake512-cp38-cp38-linux_x86_64.whl
```

#### Run a TensorFlow model on IPU

```
cd examples/tutorials/tutorials/tensorflow2/keras/completed_demos/

python completed_demo_ipu.py
```

- Deactivate the virtual environment after the model finishes running.
    ```
    deactivate
    ```

#### Monitor the IPU usage with gc-monitor command

```
watch -n 2 gc-monitor
```

#### PopTorch Virtual Environment Setup

```
cd /localdata/$USER/ipu_labs

virtualenv -p python3 poptorch_test

source poptorch_test/bin/activate

python -m pip install /opt/gc/poplar/poplar_sdk-ubuntu_20_04-3.3.0+1403-208993bbb7/poptorch-3.3.0+113432_960e9c294b_ubuntu_20_04-cp38-cp38-linux_x86_64.whl
```

#### Run a PopTorch model on IPU

```
cd examples/tutorials/simple_applications/pytorch/mnist/

pip install -r requirements.txt

python mnist_poptorch.py
```

- Deactivate the virtual environment after the model finishes running.

    ```
    deactivate
    ```

#### Monitor the IPU usage with gc-monitor command

```
watch -n 2 gc-monitor
```

### Section III. Porting TensorFlow Code to IPU

#### 1. Import the TensorFlow IPU module

Add the following import statement to the beginning of your script:

```python
from tensorflow.python import ipu
```

#### 2. Preparing the dataset

- Make sure the sizes of the datasets are divisible by the batch size
    ```python
    def make_divisible(number, divisor):
        return number - number % divisor
    ```
- Adjust dataset lengths
    ```python
    (x_train, y_train), (x_test, y_test) = load_data()
    train_data_len = x_train.shape[0]
    train_data_len = make_divisible(train_data_len, batch_size)
    x_train, y_train = x_train[:train_data_len], y_train[:train_data_len]
    test_data_len = x_test.shape[0]
    test_data_len = make_divisible(test_data_len, batch_size)
    x_test, y_test = x_test[:test_data_len], y_test[:test_data_len]
    ```

#### 3. Add IPU configuration

To use the IPU, you must create an IPU session configuration:
```
ipu_config = ipu.config.IPUConfig()
ipu_config.auto_select_ipus = 1
ipu_config.configure_ipu_system()
```

A full list of configuration options is available in the [API documentation](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/tensorflow/api.html#tensorflow.python.ipu.config.IPUConfig).

#### 4. Specify IPU strategy

```
strategy = ipu.ipu_strategy.IPUStrategy()
```

The tf.distribute.Strategy is an API to distribute training and inference across multiple devices. IPUStrategy is a subclass which targets a system with one or more IPUs attached.

#### Hands-on Session 2

- Activate the TF virtual environment
    ```
    cd /localdata/$USER/ipu_labs
    source venv_tf2/bin/activate
    ```
- Change directory to Keras
    ```
    cd IPU-Training/Keras
    ```
- Complete the #Todos in the mnist-ipu-todo.py file.
- Run it in the venv_tf2 virtual environment.
    ```
    python mnist-ipu-todo.py
    ```
- After finishing the job, you can deactivate the virtual environment
    ```
    deactivate
    ```

### Section IV. Porting PyTorch Code to IPU

#### Training a model on IPU

- Import the packages
    ```
    import torch
    import poptorch
    import torchvision
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score
    ```
#### Load the data

PopTorch offers an extension of torch.utils.data.DataLoader class with its poptorch.DataLoader class, specialized for the way the underlying PopART framework handles batching of data.

#### Build the model

```python
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
```

```python
def forward(self, x, labels=None):
    x = self.pool(self.relu(self.conv1(x)))
    x = self.norm(self.relu(self.conv2(x)))
    x = torch.flatten(x, start_dim=1)
    x = self.relu(self.fc1(x))
    x = self.log_softmax(self.fc2(x))
    # The model is responsible for the calculation of the loss when using an IPU. We do it this way:
    if self.training:
        return x, self.loss(x, labels)
    return x


model = ClassificationModel()
model.train()
```

#### Prepare training for IPUs

The compilation and execution on the IPU can be controlled using poptorch.Options. These options are used by PopTorch's wrappers such as poptorch.DataLoader and poptorch.trainingModel.

```python
opts = poptorch.Options()
train_dataloader = poptorch.DataLoader(
    opts, train_dataset, batch_size=16, shuffle=True, num_workers=20
)
```

#### Train the model

```python
optimizer = poptorch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

poptorch_model = poptorch.trainingModel(model, options=opts, optimizer=optimizer)

epochs = 30
for epoch in tqdm(range(epochs), desc="epochs"):
    total_loss = 0.0
    for data, labels in tqdm(train_dataloader, desc="batches", leave=False):
        output, loss = poptorch_model(data, labels)
        total_loss += loss

poptorch_model.detachFromDevice()

torch.save(model.state_dict(), "classifier.pth")
```

#### Evaluate the model

```python
model = model.eval()

poptorch_model_inf = poptorch.inferenceModel(model, options=opts)

test_dataloader = poptorch.DataLoader(opts, test_dataset, batch_size=32, num_workers=10)

predictions, labels = [], []
for data, label in test_dataloader:
    predictions += poptorch_model_inf(data).data.max(dim=1).indices
    labels += label

poptorch_model_inf.detachFromDevice()

print(f"Eval accuracy: {100 * accuracy_score(labels, predictions):.2f}%")
```

#### Hands-on Session 3

- Activate the PopTorch virtual environment
    ```
    cd /localdata/$USER/ipu_labs
    source poptorch_test/bin/activate
    ```
- Change directory to PyTorch
    ```
    cd IPU-Training/PyTorch
    ```
- Complete the #Todos in the fashion-mnist-pytorch-ipu-todo.py file.
- Run it in the poptorch_test virtual environment.
    ```
    pip install -r requirements.txt
    python fashion-mnist-pytorch-ipu-todo.py
    ```
- After finishing the job, you can deactivate the virtual environment
```
deactivate
```
