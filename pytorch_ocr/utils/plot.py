import os

import matplotlib.pyplot as plt
import numpy as np


def plot_losses(train_losses, valid_losses, cfg):
    plt.style.use("seaborn")
    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_losses, color="blue", label="Training loss")
    ax.plot(valid_losses, color="red", label="Validation loss")
    ax.set(title="Loss over epochs", xlabel="Epoch", ylabel="Loss")
    ax.legend()
    plt.style.use("default")
    _graph_name = os.path.join(cfg.paths.save_dir, "losses.png")
    print(f"saving losses graph at {_graph_name}")
    plt.savefig(_graph_name)


def plot_acc(accuracy, cfg):
    plt.style.use("seaborn")
    accuracy = np.array(accuracy)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(accuracy, color="purple", label="Model Accuracy")
    ax.set(title="Accuracy over epochs", xlabel="Epoch", ylabel="Accuracy")
    ax.legend()
    plt.style.use("default")
    _graph_name = os.path.join(cfg.paths.save_dir, "accuracy.png")
    print(f"saving accuracy graph at {_graph_name}")
    plt.savefig(_graph_name)
