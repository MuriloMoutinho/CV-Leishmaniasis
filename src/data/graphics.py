from matplotlib import pyplot as plt

def create_loss_history_graph(history, filename):
    epochs = range(1, len(history) + 1)

    train_loss = [h["train_loss"] for h in history]
    train_accuracy = [h["train_accuracy"] for h in history]

    plt.plot(epochs, train_loss, label="Loss treino")
    plt.plot(epochs, train_accuracy, label="Acuracia treino")

    plt.xlabel("Época")
    plt.ylabel("Métrica")
    plt.legend()
    plt.grid(True)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
