import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from torch import nn
import warnings
warnings.filterwarnings("ignore")

# Define the model architecture (must match the training one)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*34*26, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN()
model.load_state_dict(torch.load("dogs_vs_cats_model.pth", map_location=device))
model.to(device)
model.eval()

# Preprocessing function
transform = transforms.Compose([
    transforms.Resize((288, 228)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Streamlit UI
st.title("🐾 Dog vs Cat Classifier")
st.write("Upload an image and let the model decide!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width =True)

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output.item() >= 0.5)

    label = "🐶 Dog" if prediction else "🐱 Cat"
    st.markdown(f"### Prediction: **{label}**")
