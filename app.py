import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Define the model architecture (must match the training one)
class CNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.feature = nn.Sequential(
      #1
      nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='valid'),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      #2
      nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, padding='valid'),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      #3
      nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, padding='valid'),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.classifier = nn.Sequential(
      nn.Flatten(),
      #4
      nn.Linear(in_features = 128*34*34 , out_features=128),
      nn.ReLU(),
      nn.Dropout(p=0.3),
      #5
      nn.Linear(in_features=128, out_features=64),
      nn.ReLU(),
      nn.Dropout(p=0.3),
      #6
      nn.Linear(in_features=64, out_features=1),
      nn.Sigmoid()
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
    transforms.Resize((288, 288)),  # Resize to match training input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize with training mean and std
])

# Streamlit UI
st.title("🐾 Dog vs Cat Classifier")
st.write("Upload an image and let the model decide!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.round(output).item()  # Round to 0 or 1

    # Display prediction
    label = "🐶 Dog" if prediction else "🐱 Cat"
    st.markdown(f"### Prediction: **{label}**")