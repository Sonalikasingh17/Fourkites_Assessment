import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron
    
        Architecture:
        Input -&gt; FC(hidden[^0]) -&gt; ReLU -&gt; FC(hidden[^1]) -&gt; ... -&gt; FC(num_classes)
        Example:
        model = SimpleMLP(input_size=784, num_classes=10, hidden_dims=[128, 64])
    """


    def __init__(self, input_size, num_classes, hidden_dims=None):
        super(SimpleMLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_size = input_size

    for hidden_size in hidden_dims:
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(nn.ReLU())
        prev_size = hidden_size
    
    layers.append(nn.Linear(prev_size, num_classes))
    self.model = nn.Sequential(*layers)


    def forward(self, x):
        if len(x.shape) &gt; 2:
            x = x.view(x.size(0), -1)
        return self.model(x)
    

    
class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for CIFAR10

    Example:
    model = SimpleCNN(num_classes=10)
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

