import cv2
import numpy

# Inicializar video 
cap = cv2.VideoCapture("segundoIntento/pruebas.mp4")

# Obtener las propiedades del video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Verificar si se obtienen las propiedades correctamente
if frame_width <= 0 or frame_height <= 0 or fps <= 0:
    print("Error al obtener las propiedades del video. Usando valores predeterminados.")
    frame_width, frame_height, fps = 640, 480, 24

# Definir el codec y crear objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('segundoIntento/salida.avi', fourcc, fps, (frame_width, frame_height))

# Cargar el clasificador Haar Cascade para la detección de personas
player_cascade = cv2.CascadeClassifier("segundoIntento/cascade.xml") 

# Leer el primer frame
ret, frame1 = cap.read()
if not ret:
    print("No se pudo leer el primer frame.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar jugadores en el fotograma utilizando el clasificador Haar Cascade
    players = player_cascade.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=8, minSize=(50, 50))

    # Dibujar un rectángulo alrededor de cada jugador detectado
    for (x, y, w, h) in players:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Escribir el frame procesado en el video de salida
    out.write(frame)

    # Mostrar el resultado
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()