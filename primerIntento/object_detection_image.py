import cv2

#arquitectura del modelo
prototxt = "model/MobileNetSSD_deploy.prototxt.txt"
#pesos del modelo
model = "model/MobileNetSSD_deploy.caffemodel"
#clases del modelo
classes = {0:"background", 1:"aeroplane", 2:"bicycle", 3:"bird", 4:"boat",
           5:"bottle", 6:"bus", 7:"car", 8:"cat", 9:"chair",
           10:"cow", 11:"diningtable", 12:"dog", 13:"horse", 14:"motorbike",
           15:"person", 16:"pottedplant", 17:"sheep", 18:"sofa", 19:"train",
           20:"tvmonitor"}
#cargar el modelo
net = cv2.dnn.readNetFromCaffe(prototxt,model)
#leer la imagen y preprocesamiento
image =  cv2.imread("image/image1.jpg")
height, width, _ = image.shape
image_resized = cv2.resize(image, (300, 300))

#crear un blob
blob = cv2.dnn.blobFromImage(image_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))
print("blob.shape" , blob.shape)

net.setInput(blob)
detections = net.forward()

for detection in detections[0][0]:
    print(detection)

    if detection[2] > 0.30:
        label = classes[detection[1]]
        print(label)
        box = detection[3:7] * [width, height, width, height]
        x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.putText(image, "Conf: {:.2f}".format(detection[2]*100), (x_start, y_start-5), 1, 1.2, (255, 0, 0),2)
        cv2.putText(image, label, (x_start, y_start-25), 1, 1.2, (255, 0, 0),2)



cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()