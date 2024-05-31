import cv2
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import random

# Ruta del video en tu equipo
video_path = "/content/pruebas.mp4"

# Cargar el modelo y el procesador de imágenes
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

# Crear el objeto VideoCapture para leer el video
cap = cv2.VideoCapture(video_path)

# Obtener el ancho y alto del video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crear el objeto VideoWriter para guardar el video con las detecciones
output_path = "/content/salida.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

# Fuente para el texto
font = ImageFont.truetype("/content/Arial.ttf", 14)

# Diccionario para almacenar los colores de los jugadores
player_colors = {}

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

            # Asignar un color al jugador si no está asignado
            if tuple(box) not in player_colors:
                # Obtener la posición x del centro de la caja delimitadora
                center_x = (box[0] + box[2]) / 2

                # Asignar un color basado en la posición x
                if center_x <= width / 2:
                    player_colors[tuple(box)] = (255, 0, 0)  # Rojo
                else:
                    player_colors[tuple(box)] = (0, 255, 0)  # Verde

            # Dibujar el rectángulo con el color asignado
            draw.rectangle(tuple(box), outline=player_colors[tuple(box)], width=2)

            # Obtener el valor de confianza
            confidence = round(score.item(), 2)

            # Agregar el texto "persona" y el valor de confianza sobre el cuadro delimitador
            text = f"Persona: {confidence}"
            # Obtener el tamaño del texto
            text_bbox = draw.textbbox((box[0], box[1] - 15), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_position = (box[0], box[1] - text_height - 5)
            draw.rectangle([text_position, (text_position[0] + text_width, text_position[1] + text_height)],
                           fill=player_colors[tuple(box)])
            draw.text(text_position, text, fill="white", font=font)

    # Guardar el frame con las detecciones en el video de salida
    out.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

# Liberar los recursos
cap.release()
out.release()
cv2.destroyAllWindows()
