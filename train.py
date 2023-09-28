import torch
from torch import nn
from torch.utils.data import DataLoader
import snntorch.functional as SF
from spikingjelly.activation_based import learning

import sklearn.metrics as metrics

import numpy as np
import os

from load_data import get_local_tuh_dev, split_train_test
from cnn_model import CNN
from snn_model import SNN
from snn_stdp_model import ConvNetWithSTDP

import matplotlib as plt


BATCH_SIZE = 32  # 32 to start.
EPOCHS = 100  # 100 to start.
LEARNING_RATE = 5e-4  # 5e-4 to start.



def get_TUH_npy_data(filename, mmap_mode=None):
    """Gets the TUH .npy data that has been processed by Luis.

    mmap_mode is passed to np.load() to allow for memory mapping for 
    particularly large files.
    """
    data_dir = "/home/tim/SNN Seizure Detection/TUH/reshuffle"

    return np.load(os.path.join(data_dir, filename), mmap_mode=mmap_mode)

# Load the training data.
def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_dataloader

# Train the model over a single epoch.
def train_epoch(model, trainloader, loss_fn, optimizer, device):
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Calculate loss.
        predictions = model(inputs)
        targets = targets.unsqueeze(1).float()  # Add a dimension to match the output of the model.
        loss = loss_fn(predictions, targets)

        # Backpropagate error and update weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"loss: {loss.item()}")

# Train the model over all epochs.
def train_model(model, trainloader, testloader, loss_fn, optimizer, device, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_epoch(model, trainloader, loss_fn, optimizer, device)

        # Test the model after each epoch.
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Calculate outputs.
                prediction = model(inputs)

                # Calculate accuracy.
                _, predicted = torch.max(prediction.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print(f"Accuracy: {correct / total}")
        print("---------------------------")

    print("Finished training")

def print_batch_accuracy(data, targets, train=False):
    output, _ = model(data.view(BATCH_SIZE, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer(data, targets, epoch, counter, iter_counter,
        loss_hist, test_loss_hist, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

def print_epoch_accuracy(model, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Get model outputs.
            spk_rec, mem_rec = model(inputs)
            _, predicted = spk_rec.sum(dim=0).max(1)

            # Calculate accuracy.
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print(f"Accuracy: {correct / total}")

def print_epoch_rocauc(model, testloader, device):
    num_samples = len(testloader.dataset)
    labels = np.empty((num_samples))
    predictions = np.empty((num_samples))

    num_steps = 23  # NOTE: This is hardcoded for now.
    batch_no = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Get model outputs.
            spk_rec, mem_rec = model(inputs)
            # Calculate the probability estimate of the positive class.
            # This is done by getting the ratio of positive spikes to the total 
            # number of spikes.
            spk_sum = spk_rec.sum(dim=0)
            predicted = spk_sum[:, 1] / spk_sum.sum(dim=1)
            predicted[np.isnan(predicted)] = 0

            # Add to the labels and predictions lists.
            labels[batch_no*BATCH_SIZE:(batch_no+1)*BATCH_SIZE] = targets.detach().cpu().numpy()
            predictions[batch_no*BATCH_SIZE:(batch_no+1)*BATCH_SIZE] = predicted.detach().cpu().numpy()

            batch_no += 1

    # Calculate the ROC AUC.
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
    auc = metrics.auc(fpr, tpr)
    print(f"ROC AUC: {auc}")

def train_snn_model(model, trainloader, testloader, loss_fn, optimizer, device, epochs):
    train_loss_hist = []
    test_loss_hist = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        minibatch_ctr = 1
        train_batch = iter(trainloader)

        for inputs, targets in train_batch:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass.
            model.train()
            spk_rec, mem_rec = model(inputs)
            # # Print something if there are any spikes at all.  # TODO: Remove this.
            # if spk_rec.sum() > 0:
            #     print(f"Spikes in minibatch {minibatch_ctr}: {spk_rec.sum()}")

            # Initialise the loss and sum over time.
            # loss_val = torch.zeros((1), dtype=torch.float, device=device)
            loss_val = loss_fn(mem_rec, targets)
            print(f"Training Loss (minibatch {minibatch_ctr}): {loss_val.item():.3f}")

            # Gradient calculation and weight update.
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting.
            train_loss_hist.append(loss_val.item())

            # # Test the model after each minibatch.
            # with torch.no_grad():
            #     model.eval()
            #     test_data, test_targets = next(iter(testloader))
            #     test_data, test_targets = test_data.to(device), test_targets.to(device)

            #     # Test set forward pass.
            #     test_spk, test_mem = model(test_data)

            #     # Test set loss.
            #     test_loss = loss_fn(test_mem, test_targets)
            #     test_loss_hist.append(test_loss.item())

            #     # Print train/test loss/accuracy:
            #     if counter % 100 == 0:
            #         print(f"Train Set Loss: {train_loss_hist[counter]:.2f}")
            #         print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
            #         # print_batch_accuracy(inputs, targets, train=True)
            #         # print_batch_accuracy(test_data, test_targets, train=False)
            #         print("\n")
            minibatch_ctr += 1

        with torch.no_grad():
            print_epoch_accuracy(model, testloader, device)
            print_epoch_rocauc(model, testloader, device)

    print("\n ===== Finished training =====")

def train_snn_stdp_model(model, trainloader, testloader, loss_fn, learner, optimizer, device, epochs):
    train_loss_hist = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        minibatch_ctr = 1
        train_batch = iter(trainloader)

        for inputs, targets in train_batch:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass.
            model.train()
            spk_rec = model(inputs)
            _, predictions = spk_rec.sum(dim=0).max(1)
            predictions = predictions.float()

            # Calculate loss.
            loss_val = loss_fn(predictions, targets)
            print(f"Training Loss (minibatch {minibatch_ctr}): {loss_val.item():.3f}")

            # Gradient calculation and weight update.
            optimizer.zero_grad()
            learner.step(on_grad=True)
            optimizer.step()

            # Store loss history for future plotting.
            train_loss_hist.append(loss_val.item())

            # Print accuracy.
            acc = np.mean((targets == predictions).detach().cpu().numpy())
            print(f"Train set accuracy for minibatch {minibatch_ctr}: {acc*100:.2f}%")

            minibatch_ctr += 1

    print("\n ===== Finished training =====")

def f_weight(x):
    return torch.clamp(x, -1, 1.)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    # Load the training data.
    # trainx = get_TUH_npy_data("train_data.npy", mmap_mode="r")
    # trainy = get_TUH_npy_data("trainy.npy")
    trainx, trainy = get_local_tuh_dev(sz_ratio=0.3, file_count=10000, mmap_mode='r') 
    # Calculate the percentage of seizure samples in the dataset.
    print(f"Training Dataset Seizure Ratio: {np.count_nonzero(trainy)/len(trainy)}")
    train_data = list(zip(trainx, trainy))
    trainloader = create_data_loader(train_data, BATCH_SIZE)

    # Load the test data.
    testx, testy = get_local_tuh_dev(sz_ratio=None, file_count=2000, mmap_mode='r')
    # Calculate the percentage of seizure samples in the dataset.
    print(f"Test Dataset Seizure Ratio: {np.count_nonzero(testy)/len(testy)}")
    test_data = list(zip(testx, testy))
    testloader = create_data_loader(test_data, BATCH_SIZE)

    # Create the model.
    # model = CNN().to(device)
    model = SNN().to(device)
    # model = ConvNetWithSTDP().to(device)

    # tau_pre, tau_post = 2., 2.
    # learner = learning.STDPLearner(step_mode='s', synapse=model.conv1, sn=model.neuron1,
    #                                tau_pre=tau_pre, tau_post=tau_post,
    #                                f_pre=f_weight, f_post=f_weight)

    # Initialise the loss function and optimizer.
    # loss_fn = nn.BCELoss()
    loss_fn = SF.mse_membrane_loss()
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model.
    # train_model(model, trainloader, testloader, loss_fn, optimizer, device, EPOCHS)
    train_snn_model(model, trainloader, testloader, loss_fn, optimizer, device, EPOCHS)
    # train_snn_stdp_model(model, trainloader, testloader, loss_fn, learner, optimizer, device, EPOCHS)

    # Save the model.
    torch.save(model.state_dict(), "cnn_model.pth")
    print("Saved PyTorch Model State to model.pth")