import cv2

#arquitectura del modelo
prototxt = "model/MobileNetSSD_deploy.prototxt.txt"
#pesos del modelo
model = "model/MobileNetSSD_deploy.caffemodel"
#clases del modelo
classes = {0:"person"}
#cargar el modelo
net = cv2.dnn.readNetFromCaffe(prototxt,model)
#leer la imagen y preprocesamiento
image =  cv2.imread("image/image.jpg")
height, width, _ = image.shape
image_resized = cv2.resize(image, (300, 300))

#crear un blob
blob = cv2.dnn.blobFromImage(image_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))
print("blob.shape" , blob.shape)

cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()