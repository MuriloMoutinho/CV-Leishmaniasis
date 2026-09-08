import streamlit as st
import torch
from pathlib import Path
from PIL import Image
from augmentation import get_images_transformations

image_transformations = get_images_transformations()['val']

MODEL_PATH = Path(__file__).resolve().parent / "results" / "resnet_aug.pt"
modelo = torch.load(MODEL_PATH, weights_only=False)

def test_model(model, test_image):
    transform = image_transformations

    test_image_tensor = transform(test_image)
    test_image_tensor = test_image_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.cuda()

    # Não precisa atualizar os coeficientes do modelo
    with torch.no_grad():
        model.eval()

        result = model(test_image_tensor)

        probability  = torch.sigmoid(result) # Transforma o logit bruto em probabilidade (0.0 a 1.0)
        prediction = (probability >= 0.5).int().item() # Decide a classe final: se >= 0.5 vira 1, senão vira 0

    return prediction

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
        st.sidebar.write("Suba uma imagem")
    else:
        u_img = Image.open(uploaded_file)

        with st.spinner('Carregando'):

            out = test_model(modelo, u_img)
            result_label = "Negativo" if out == 0 else "Positivo"
            st.success(result_label)