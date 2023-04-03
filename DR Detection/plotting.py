import matplotlib.pyplot as plt #For plotting.

def loss_plots(epochs, train_losses, valid_losses):
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    plt.plot(epochs, valid_losses, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()