import cv2
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import random

# Verificar si hay una GPU disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ruta del video en tu equipo
video_path = "/content/pruebas.mp4"

# Cargar el modelo y el procesador de imágenes y mover el modelo a la GPU
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny').to(device)
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

    with torch.no_grad():
        # Procesar la imagen con el modelo de detección de objetos
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

        # Obtener las detecciones de personas y raquetas de tenis
        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = image_processor.post_process_object_detection(outputs, threshold=0.85, target_sizes=target_sizes)[0]

    # Dibujar las cajas delimitadoras en el frame
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if model.config.id2label[label.item()] == "person":
            box = [round(i, 2) for i in box.tolist()]

            # Obtener la posición x del centro de la caja delimitadora
            center_x = (box[0] + box[2]) / 2

            # Buscar la raqueta de tenis más cercana
            tennis_racket_box = None
            for racket_score, racket_label, racket_box in zip(results["scores"], results["labels"], results["boxes"]):
                if model.config.id2label[racket_label.item()] == "tennis racket":
                    tennis_racket_box = racket_box
                    break

            # Si se encuentra una raqueta de tenis, colorear al jugador según su posición
            if tennis_racket_box is not None:
                # Obtener la posición x del centro de la raqueta de tenis
                tennis_racket_center_x = (tennis_racket_box[0] + tennis_racket_box[2]) / 2

                # Asignar color según la posición relativa a la raqueta
                if center_x < tennis_racket_center_x:
                    player_colors[tuple(box)] = (255, 0, 0)  # Rojo (izquierda)
                else:
                    player_colors[tuple(box)] = (0, 255, 0)  # Verde (derecha)

            # Dibujar el rectángulo con el color asignado
            color = player_colors.get(tuple(box), (0, 0, 0))  # Default color is black if key is not found
            draw.rectangle(tuple(box), outline=color, width=2)

            # Obtener el valor de confianza
            confidence = round(score.item(), 2)

            # Agregar el texto "persona" y el valor de confianza sobre el cuadro delimitador
            text = f"Persona: {confidence}"
            # Obtener el tamaño del texto
            text_bbox = draw.textbbox((box[0], box[1] - 15), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_position = (box[0], box[1] - text_height - 5)
            draw.text(text_position, text, fill="white", font=font)

    # Guardar el frame con las detecciones en el video de salida
    out.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    # Liberar memoria de tensores no utilizados
    del inputs, outputs, target_sizes, results
    torch.cuda.empty_cache()

# Liberar los recursos
cap.release()
out.release()
cv2.destroyAllWindows()