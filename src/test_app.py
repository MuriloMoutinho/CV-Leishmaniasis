import numpy
import streamlit as st
import torch
from pathlib import Path
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from config import create_binary_model, create_augmentation

PROJECT_PATH = Path(__file__).resolve().parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = PROJECT_PATH / "experiments" / "models" / "total" / "oficial" / "DLB_resnet50_strong.pth"
model = create_binary_model("resnet50")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

visual_aug_test = "strong"
test_transform = create_augmentation("weak")["val"]
strong_transform = create_augmentation(visual_aug_test)["train"]

MEAN = numpy.array([0.485, 0.456, 0.406])
STD = numpy.array([0.229, 0.224, 0.225])


@st.cache_resource
def load_model():
    model = create_binary_model("resnet50")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()


def predict_with_cam(model, pil_image, target_layer, explain_class=None):
    tensor = test_transform(pil_image).unsqueeze(0).to(device)

    # Predição sem gradiente
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor)).item()
    prediction = int(probability >= 0.5)

    # Grad-CAM precisa de gradiente, então fora do no_grad
    category = prediction if explain_class is None else explain_class

    cam_extractor = GradCAM(model=model, target_layers=[target_layer])
    try:
        grayscale_cam = cam_extractor(
            input_tensor=tensor,
            targets=[BinaryClassifierOutputTarget(category)],
        )[0]
    finally:
        cam_extractor.activations_and_grads.release()  # remove os hooks

    # Desfaz a normalização para ter a imagem RGB em [0, 1]
    rgb = tensor[0].cpu().numpy().transpose(1, 2, 0)
    rgb = numpy.clip(rgb * STD + MEAN, 0, 1).astype(numpy.float32)

    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    return prediction, probability, overlay

def apply_strong_augmentation(pil_image):
    augmented = strong_transform(pil_image)

    # Desfaz a normalizacao
    image = augmented.cpu().numpy().transpose(1, 2, 0)
    image = image * STD + MEAN
    image = numpy.clip(image, 0, 1)

    # [0, 1] -> [0, 255]
    image = (image * 255).astype(numpy.uint8)

    return Image.fromarray(image)

model = load_model()
for param in model.layer4.parameters():
    param.requires_grad = True

target_layer = model.layer4[-1].conv3

st.title("Lesh Pytorch")
st.write("\n")

st.sidebar.title("Suba uma imagem")
uploaded_file = st.sidebar.file_uploader(" ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(Image.open(uploaded_file), width="stretch")

st.sidebar.write("\n")

if st.sidebar.button("Enviar"):
    if uploaded_file is None:
        st.sidebar.write("Suba uma imagem")
    else:
        u_img = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Carregando"):
            prediction, prob, overlay = predict_with_cam(model, u_img, target_layer)

        result_label = "Negativo" if prediction == 0 else "Positivo"
        st.success(f"{result_label} (prob. positivo: {prob:.2%})")

        st.subheader(f"Teste Augmentation {visual_aug_test}")
        st.image(
            apply_strong_augmentation(u_img),
            caption="Imagem após Augmentation",
            width="stretch"
        )

        st.subheader("Grad-CAM")
        st.image(overlay, caption="Regiões que mais influenciaram a predição", width="stretch")