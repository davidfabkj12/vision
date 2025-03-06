import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image

# Charger le modèle YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Utilisation du GPU si disponible

# Fonction pour effectuer la détection sur une image
def detect_objects(image):
    results = model(image)
    detections = results.xyxy[0].cpu().numpy()  # Extraction des résultats
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = f"{model.names[int(cls)]} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Interface Streamlit
st.title("Détection d'objets en temps réel avec YOLOv5")
run = st.checkbox('Démarrer la détection')

# Accès à la webcam
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Erreur lors de la capture de l'image.")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = detect_objects(frame)
    FRAME_WINDOW.image(frame)
else:
    st.write('Arrêt de la détection.')

cap.release()
