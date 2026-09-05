import torch, time, os, numpy, PIL.Image
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optmin

dataset_path = r'test/'
folder_train = os.path.join(dataset_path, 'train')
folder_val = os.path.join(dataset_path, 'val')
folder_test = os.path.join(dataset_path, 'test')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def get_images_transformations():
    return {
        'train': transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
        ])
    }

def get_transformed_dataset(image_transforms):
    return {
        'train': datasets.ImageFolder(root=folder_train, transform=image_transforms['train']),
        'val': datasets.ImageFolder(root=folder_val, transform=image_transforms['val']),
        #'test': datasets.ImageFolder(root=folder_test, transform=image_transforms['test']),
    }

def create_cnn_model():
    cnn_model = models.alexnet(pretrained=True)

    for param in cnn_model.parameters():
        param.requires_grad = False
    # trava o treinamento da rede

    classes_number = len(os.listdir(folder_train))  # determinado pela quantidade de pastas (2 neste caso, positivo e negativo)
    #classes_hash = {v: k for k, v in images['train'].class_to_idx.items()}

    cnn_model.classifier[6] = nn.Linear(4096, classes_number)  # numero de neuronios. Essa linha substitui a ultima camada 6
    #numero de classes define que a rede neural irá acabar em 2 nós (pois tem 2 classes)

    cnn_model.classifier.add_module("7", nn.LogSoftmax(dim=1))  # adiciona um bloco camada de softmax. Essa linha adiciona uma nova camada 7
    # ao adicionar 2 novas camadas e treinar, apenas essas novas camadas não travadas vão ser afetadas pelo treinamento

    cnn_model.to(device)
    return cnn_model

#dividir função train e validate, receber lista de imagens por parametro
def train_and_validate(model, error_function, optimizer, learning_rate=0.001, batch_size=32, epoch_num=25):
    train_data_loader = DataLoader(images['train'], batch_size=batch_size, shuffle=True)
    train_data_val = DataLoader(images['val'], batch_size=batch_size, shuffle=True)

    num_train_images = len(images['train'])
    num_val_images = len(images['train'])

    history = []
    best_accuracy = 0

    for epoch in range(epoch_num):
        start = time.time()
        print(f"Época: {epoch + 1}/{epoch_num}")

        # vai definir como modo de treino, assim afetando as camadas da rede que não estão paralizadas
        model.train()

        error_train = 0.0
        accuracy_train = 0.0

        error_val = 0.0
        accuracy_val = 0.0

        #itera para cada lote de imagem, os batchs sao os tensores do lote
        #labels sao as classificacoes de cada imagem do lote
        for i_batch, (image_batch, label_class) in enumerate(train_data_loader):

            #define cpu ou gpu
            image_batch = image_batch.to(device)
            label_class = label_class.to(device)

            #zera gradiente, ia faz predicao, calcula loss, faz backward, atualiza pessos ia
            optimizer.zero_grad()
            results = model(image_batch)
            error = error_function(results, label_class)
            error.backward()
            optimizer.step()

            error_train += error.item() * image_batch.size(0) #erro total do lote, e soma o erro no treino

            max_results, indices = torch.max(results, 1)
            corrects_results = indices.eq(label_class.data.view_as(indices))

            accuracy = torch.mean(corrects_results.type(torch.FloatTensor))
            accuracy_train += accuracy.item() * image_batch.size(0)

            print(f" Treino - lote {i_batch}, erro {error.item()}, acuracia: {accuracy.item()}")

        with torch.no_grad():

            model.eval() #modo de treino

            for i_batch, (image_batch, label_class) in enumerate(train_data_val):
                image_batch = image_batch.to(device)
                label_class = label_class.to(device)

                results = model(image_batch)  # calcula a saida da imagem usando o modelo

                error = error_function(results, label_class)  # verifica o resultado com a label correta
                error_val += error.item() * image_batch.size(0)  # erro total do lote, e soma o erro no treino

                max_results, indices = torch.max(results, 1)
                corrects_results = indices.eq(label_class.data.view_as(indices))

                accuracy = torch.mean(corrects_results.type(torch.FloatTensor))
                accuracy_val += accuracy.item() * image_batch.size(0)

                print(f" Validação - lote {i_batch}, erro {error.item()}, acuracia: {accuracy.item()}")

        average_error_train = error_train / num_train_images
        average_accuracy_train = accuracy_train / num_train_images

        average_error_val = error_val / num_val_images
        average_accuracy_val = accuracy_val / num_val_images

        history.append([average_error_train, average_accuracy_train, average_error_val, average_accuracy_val])

        end = time.time()

        print(
            f"Época : {epoch + 1}, Tempo: {end - start}s"
            f"\n\t\tTreino: Erro: {average_error_train}, Acurácia: {average_accuracy_train*100}%, "
            f"\n\t\tValidação : Erro : {average_error_val}, Acurácia: {average_accuracy_val*100}%, "
        )

        if average_accuracy_val > best_accuracy:
            best_accuracy = average_accuracy_val
            #torch.save(model.state_dict(), "models/melhor_modelo.pth")
            torch.save(model, 'models/melhor_modelo.pt')

    return history


image_size = 224
images = get_transformed_dataset(get_images_transformations())

#estimativa do gradiente mais "ruidosa" para batchs menores. Batchs maiores tras uma média melhor, mas pode piorar generalização
batch_size = 10 #vai pegar 10 imagens por vez do dataset e enviá-las para a rede.
epoch_num = 20
learning_rate = 0.001 #verificar linear scaling rule

alexnet = create_cnn_model()
error_function = nn.BCEWithLogitsLoss()
optimizer = optmin.AdamW(alexnet.parameters())

model_history = train_and_validate(alexnet, error_function, optimizer, learning_rate, batch_size, epoch_num)

np_model_history = numpy.array(model_history)
plt.plot(np_model_history[:, 0:2])
plt.legend(['Erro treino', 'Erro validação'])
plt.xlabel("Época")
plt.ylabel("Erro")
plt.ylim(0,0.5)
plt.show()
