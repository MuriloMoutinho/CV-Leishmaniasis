import torch

#estimativa do gradiente mais "ruidosa" para batchs menores. Batchs maiores tras uma média melhor, mas pode piorar generalização
batch_size = 16 #vai pegar x imagens por vez do dataset e enviá-las para a rede.
epoch_num = 50
learning_rate = 0.001 #verificar linear scaling rule (valor padrão)
weight_decay = 0.01

def create_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

def create_loss_function():
    return torch.nn.BCEWithLogitsLoss()