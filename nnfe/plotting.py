
import matplotlib.pyplot as plt
import numpy as onp
import jax

def plot_loss(loss_vec, results_dir):
    fig, ax = plt.subplots()
    ax.semilogy(onp.arange(loss_vec.shape[0]), loss_vec)
    ax.set_title("Loss vs. Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.savefig(results_dir + "/plots/loss.png")
    
    return


def plot_grad():

    return

def plot_learning_rate(scheduler, epochs, results_dir):
    fig, ax = plt.subplots()
    ax.semilogy(onp.arange(epochs), jax.vmap(scheduler)(onp.arange(epochs)))
    ax.set_title("Learning rate vs. Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    plt.savefig(results_dir + "/plots/LR_plot.png")
    
    return

