import streamlit as st
import base64
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from transformers import AutoProcessor, Owlv2ForObjectDetection, Owlv2Processor
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import torchvision.transforms.functional as TF

load_dotenv()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def GPT4(input_image):
    
    base64_image = encode_image(input_image)

    client = OpenAI()

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Do you detect rust on this picture? Or do you see another anomaly?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 400
    }

    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    choices = response.json()['choices']
    message = choices[0]['message']
    content = message['content']
    return content

def GPT4o(input_image):
    
    base64_image = encode_image(input_image)

    client = OpenAI()

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Do you detect rust on this picture? Or do you see another anomaly?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 400
    }

    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    choices = response.json()['choices']
    message = choices[0]['message']
    content = message['content']
    return content


def clip(input_image, sensitivity):
# Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_image = Image.open(input_image)

    processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-finetuned")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-finetuned").to(device)

    texts = [["corrosion or rust", "birds nest"]]
    inputs = processor(text=texts, images=input_image, return_tensors="pt").to(device)
    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    def get_preprocessed_image(pixel_values):
        pixel_values = pixel_values.squeeze().numpy()
        unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        unnormalized_image = Image.fromarray(unnormalized_image)
        return unnormalized_image

    unnormalized_image = get_preprocessed_image(inputs.pixel_values.cpu())
    target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
    results = processor.post_process_object_detection(
        outputs=outputs, threshold=sensitivity, target_sizes=target_sizes
    )

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    draw = ImageDraw.Draw(unnormalized_image)

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        st.write(f"Detected {text[label]} with confidence {round(score.item(), 3)}")
        draw.rectangle(box, outline="red", width=4)
        category = 'Cat:' + str(label.item())
        font = ImageFont.truetype("arial.ttf", size=15)
        text_position = (box[0]+10, box[3]-40)
        draw.text(text_position, category, fill="red", font=font)
    
    new_size = (512, 512)
    resized_image = unnormalized_image.resize(new_size)

    return resized_image


def resnet(input_image, sensitivity_resnet):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    num_classes = 5  
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load('model/model_weights.pth'))

    # Load the image
    img = Image.open(input_image).convert('RGB')

    # Convert the PIL Image to a PyTorch Tensor
    # Instead of manually converting and normalizing, let's use torchvision transforms
    transform = T.Compose([ T.ToTensor() ])

    img_tensor = transform(img)

    # Add a batch dimension since PyTorch models expect batches
    img_tensor = img_tensor.unsqueeze(0)

    # Determine the device dynamically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)  # Move tensor to the appropriate device

    # Ensure the model is on the same device and set it to evaluation mode
    model = model.to(device)
    model.eval()

    # Perform the prediction
    with torch.no_grad():
        prediction = model(img_tensor)

    # Process the prediction output as needed
    im = TF.to_pil_image(img_tensor.squeeze().cpu())
    draw = ImageDraw.Draw(im)
    st.write(f"Detected {prediction[0]['scores']}")

    for index, box in enumerate(prediction[0]['boxes'].cpu().numpy()):
        if prediction[0]['scores'][index] > sensitivity_resnet:
            draw.rectangle(box, width=3, outline="red")
            text = str(prediction[0]['labels'][index].item())
            text = text + ' score: ' + str(round(prediction[0]['scores'][index].item(),2))
            font = ImageFont.truetype("arial.ttf", size=10)
            text_position = (box[0], box[3])
            draw.text(text_position, text, fill="red", font=font)

    return im


################################################################
# GUI
################################################################
st.set_page_config(layout="wide")
st.title("High Voltage Tower Monitor")

with st.sidebar:
   st.image('D:/Projects/Tower/elia.png')

   st.subheader("Upload Image")
   image = st.file_uploader('Chose file')
   
   st.subheader("Choose Model")
   check1 = st.checkbox('ResNet')
   slider2 = st.slider('Resnet Sesitivity:', value=80)
   check3 = st.checkbox('CLIP')
   slider1 = st.slider('CLIP Sesitivity:', value=8)
   check2 = st.checkbox('GPT4-Turbo')
   check3 = st.checkbox('GPT4o')

   button = st.button("Submit")


if button:
    col1 , col2 = st.columns(2)

    with col1:
        if image:
            st.subheader("Original Image")
            st.image(image)
            image_path = f"D:/Projects/Tower3/{image.name}"
    with col2:
        if check1:
            st.subheader("ResNet")
            st.image(resnet(image_path, slider2/100))
        if check3:
            st.subheader("CLIP") 
            st.image(clip(image_path, slider1/100))
    with col1:
        if check2:
            st.subheader("GPT4-Turbo")
            st.write(GPT4(image_path))
        if check3:
            st.subheader("GPT4o")
            st.write(GPT4o(image_path))
