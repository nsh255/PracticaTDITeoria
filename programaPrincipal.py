import cv2
import subprocess

# Cargar el video
cap = cv2.VideoCapture('pruebas.mp4')

# Verificar si el video se cargó correctamente
if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

# Cargar el clasificador Haar Cascade para la detección de personas
player_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Definir el codec y crear un objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('salida.mp4', fourcc, 20.0, (640, 480))

# Leer y procesar cada fotograma
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar jugadores en el fotograma utilizando el clasificador Haar Cascade
    players = player_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(40, 40))    
    # Dibujar un rectángulo alrededor de cada jugador detectado
    for (x, y, w, h) in players:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    # Escribir el fotograma en el nuevo archivo de video
    out.write(frame)

# Liberar el video y cerrar las ventanas
cap.release()
out.release()
cv2.destroyAllWindows()