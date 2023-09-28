import torch
from torch import nn
import snntorch as snn
from snntorch import surrogate
from torchsummary import summary

class SNN(nn.Module):
    def __init__(self):
        super().__init__()

        spike_grad_lstm = surrogate.straight_through_estimator()
        spike_grad_fc = surrogate.fast_sigmoid(slope=97)
        self.thr = 0.10315190601276118
        beta = 0.5817305509700217
        p1 = 0.35638768935049997
        p2 = 0.23176980850656767

        # Initialise layers.
        self.sclstm1 = snn.SConv2dLSTM(
            in_channels=1,
            out_channels=16,
            kernel_size=(19,3),
            max_pool=2,
            spike_grad=spike_grad_lstm,
            threshold = self.thr
        )
        self.sclstm2 = snn.SConv2dLSTM(
            in_channels=16,
            out_channels=64,
            kernel_size=3,
            max_pool=2,
            spike_grad=spike_grad_lstm,
            threshold = self.thr
        )
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(p1)
        self.fc1 = nn.Linear(64 * 124, 128)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad_fc, reset_mechanism='subtract',threshold = self.thr)
        self.dropout2 = nn.Dropout(p2)
        self.fc2 = nn.Linear(128,2)
        self.lif2 = snn.Leaky(beta=0.5, spike_grad=spike_grad_fc, reset_mechanism='subtract',threshold = self.thr)

    def forward(self, x):
        # Initialise hidden states and outputs at t=0.
        syn1, mem1 = self.sclstm1.init_sconv2dlstm()
        syn2, mem2 = self.sclstm2.init_sconv2dlstm()
        mem3 = self.lif1.init_leaky()
        mem4 = self.lif2.init_leaky()

        # Record the final layer
        spk4_rec = []
        mem4_rec = []

        # Number of steps assuming x is [N, T, C, H, W] with 
        # N = Batches, T = Time Steps, C = Channels, H = Height, W = Width
        # num_steps = x.size()[1]
        num_steps = 23

        for step in range(num_steps):
            x_step = x[:, step, :, :, :]
            spk1, syn1, mem1 = self.sclstm1(x_step, syn1, mem1)
            spk2, syn2, mem2 = self.sclstm2(spk1, syn2, mem2)
            spk2 = self.flatten(spk2)
            spk2 = self.dropout1(spk2)
            cur3 = self.fc1(spk2)
            spk3, mem3 = self.lif1(cur3, mem3)
            spk3 = self.dropout2(spk3)
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif2(cur4, mem4)

            spk4_rec.append(spk4)
            mem4_rec.append(mem4)

        return torch.stack(spk4_rec, dim=0), torch.stack(mem4_rec, dim=0)



if __name__ == "__main__":
    model = SNN()
    print(model)
    summary(model, input_size=(23, 1, 19, 125))