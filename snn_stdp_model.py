import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, learning, functional, surrogate
from torchsummary import summary

class ConvNetWithSTDP(nn.Module):
    def __init__(self):
        super().__init__()

        # 2x 1D convolutional layers with max pooling.
        self.conv1 = nn.Sequential(
            layer.Conv2d(in_channels=1,
                         out_channels=8,
                         kernel_size=3
            ),
            layer.BatchNorm2d(num_features=8),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=2, stride=1)
        )
        self.conv2 = nn.Sequential(
            layer.Conv2d(in_channels=8,
                         out_channels=64,
                         kernel_size=3
            ),
            layer.BatchNorm2d(num_features=64),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=2, stride=1)
        )

        # 2x fully connected layers.
        self.flatten = layer.Flatten()
        self.dropout1 = layer.Dropout(0.5)
        self.fc1 = layer.Linear(64 * 1547, 128)
        self.neuron1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.dropout2 = layer.Dropout(0.5)
        self.fc2 = layer.Linear(128, 2)
        self.neuron2 = neuron.IFNode(surrogate_function=surrogate.ATan())

    def forward(self, x):
        # x.shape = [N, T, C, H, W]
        # N = Batches, T = Time Steps, C = Input Channels, 
        # H = Electrode Channels (Height), W = Frequency Bins (Width)
        num_steps = 23

        # Record the final layer
        spk_rec = []

        for step in range(num_steps):
            x_step = x[:, step, :, :, :]
            x_step = self.conv1(x_step)
            x_step = self.conv2(x_step)
            x_step = self.flatten(x_step)
            x_step = self.dropout1(x_step)
            x_step = self.fc1(x_step)
            x_step = self.neuron1(x_step)
            x_step = self.dropout2(x_step)
            x_step = self.fc2(x_step)
            out_spikes = self.neuron2(x_step)

            spk_rec.append(out_spikes)

        return torch.stack(spk_rec, dim=0)

if __name__ == "__main__":
    model = ConvNetWithSTDP()
    print(model)
    summary(model, input_size=(23, 1, 19, 125))