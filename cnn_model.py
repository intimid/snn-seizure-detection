from torch import nn
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 1-D convolutional layers with max pooling.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=19,
                      out_channels=32,
                      kernel_size=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.batchnorm2 = nn.BatchNorm2d(64)
        # 2 fully connected layers.
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 62, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        # Sigmoid activation function.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        logits = self.fc3(x)
        predictions = self.sigmoid(logits)
        return predictions



if __name__ == "__main__":
    model = CNN()
    print(model)
    summary(model, input_size=(19, 125, 23))