import cv2
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw
import torch
import numpy as np

# Ruta del video en tu equipo
video_path = "/path/to/video.mp4"

# Cargar el modelo y el procesador de imágenes
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

# Crear el objeto VideoCapture para leer el video
cap = cv2.VideoCapture(video_path)

# Obtener el ancho y alto del video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crear el objeto VideoWriter para guardar el video con las detecciones
output_path = "/path/to/output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

while cap.isOpened():
    # Leer el siguiente frame del video
    ret, frame = cap.read()

    if not ret:
        break

    # Convertir el frame a formato PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Procesar la imagen con el modelo de detección de objetos
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Obtener las detecciones de personas
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.85, target_sizes=target_sizes)[0]

    # Dibujar las cajas delimitadoras en el frame
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if model.config.id2label[label.item()] == "person":
            box = [round(i, 2) for i in box.tolist()]
            draw.rectangle(tuple(box), outline="red", width=2)

    # Guardar el frame con las detecciones en el video de salida
    out.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

# Liberar los recursos
cap.release()
out.release()
cv2.destroyAllWindows()
