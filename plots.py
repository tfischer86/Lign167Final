import numpy as np
import matplotlib.pyplot as plt

def make_training_plot(sparse, dense, title, xlabel, ylabel, filename):
    fig = plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    def plot(train, val, c, label):
        plt.plot(train, c + '--', label=label + ' train')
        plt.plot(val, c, label=label + ' validation')

    plot(*sparse, 'r', 'sparse')
    plot(*dense, 'b', 'sparse_frozen')

    plt.legend()
    plt.savefig(filename)


if __name__ == "__main__":
   sparse_train_loss = np.load("sparse_train_loss.npy")
   sparse_train_acc  = np.load("sparse_train_acc.npy")
   sparse_val_loss   = np.load("sparse_val_loss.npy")
   sparse_val_acc    = np.load("sparse_val_acc.npy")

   dense_train_loss = np.load("dense_train_loss.npy")
   dense_train_acc  = np.load("dense_train_acc.npy")
   dense_val_loss   = np.load("dense_val_loss.npy")
   dense_val_acc    = np.load("dense_val_acc.npy")

   make_training_plot((sparse_train_loss, sparse_val_loss), (dense_train_loss, dense_val_loss), "Model Losses", "Epochs", "Cross Entropy Loss", "losses.png")
   make_training_plot((sparse_train_acc, sparse_val_acc), (dense_train_acc, dense_val_acc), "Model Accuracies", "Epochs", "Accuracy (%)", "accuracies.png")

   #sparse_frozen_train_loss = np.load("sparse_frozen_train_loss.npy")
   #sparse_frozen_train_acc  = np.load("sparse_frozen_train_acc.npy")
   #sparse_frozen_val_loss   = np.load("sparse_frozen_val_loss.npy")
   #sparse_frozen_val_acc    = np.load("sparse_frozen_val_acc.npy")
   #make_training_plot((sparse_train_loss, sparse_val_loss), (sparse_frozen_train_loss, sparse_frozen_val_loss), "Model Losses", "Epochs", "Cross Entropy Loss", "sparse_vs_frozen_losses.png")
   #make_training_plot((sparse_train_acc, sparse_val_acc), (sparse_frozen_train_acc, sparse_frozen_val_acc), "Model Accuracies", "Epochs", "Cross Entropy Loss", "sparse_vs_frozen_accuracies.png")
