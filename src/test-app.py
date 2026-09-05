import streamlit as st
import time, torch
from pathlib import Path
from PIL import Image
from torchvision import transforms


image_size = 100

image_transformations = transforms.Compose([
        transforms.Resize(size=[image_size, image_size]),
        transforms.ToTensor(),
    ])

MODEL_PATH = Path(__file__).resolve().parent / "models" / "melhor_modelo.pt"
modelo = torch.load(MODEL_PATH, weights_only=False)


def test_model(model, test_image):
    transform = image_transformations

    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, image_size, image_size).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, image_size, image_size)

    # Não precisa atualizar os coeficientes do modelo
    with torch.no_grad():
        model.eval()

        # Modelo retorna as probabilidades em log (log softmax)
        result = model(test_image_tensor)

        # torch.exp para voltar a probabilidade de log para a probabilidade linear
        ps = torch.exp(result)

        # topk retorna o os k maiores valores do tensor
        # o tensor de probabilidades vai trazer na 1a posição a classe com maior
        # probabilidade de predição
        number_classes = 2
        topk, topclass = ps.topk(number_classes, dim=1)

    return topclass[0][0]

st.title("Lesh Pytorch")
st.write('\n')

st.sidebar.title("Suba uma imagem")

uploaded_file = st.sidebar.file_uploader(" ", type=['jpg', 'jpeg'])

if uploaded_file is not None:
    u_img = Image.open(uploaded_file)
    st.image(u_img, width="stretch")

st.sidebar.write('\n')


if st.sidebar.button("Enviar"):
    if uploaded_file is None:
        u_img = Image.open(uploaded_file)
        st.image(u_img, width="stretch")
        st.sidebar.write("Suba uma imagem")

    else:

        with st.spinner('Carregando'):

            prediction = test_model(modelo, u_img)
            time.sleep(2)
            st.success('Pronto!')

        print(prediction)

        if prediction == 0:
            st.sidebar.write("Negativo", '\n')
        elif prediction == 1:
            st.sidebar.write("Positivo", '\n')
