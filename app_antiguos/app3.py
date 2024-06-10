import sys
import cv2
import imutils
from yoloDet import YoloTRT
import requests
import tempfile
import io
from threading import Thread

# use path for library and engine file
engines = [ 'yolov7/build/best.engine', 'yolov7/build/bestYOLOv7Prueba2.engine']
model = YoloTRT(library="yolov7/build/libmyplugins.so", engine=engines[1], conf=0.9, yolo_ver="v7")
chat_id = '1045458604'
vids = [ 'videos/testvideo.mp4', 'video2.mp4' , 'armalarga1.mp4', 'armalarga2.mp4']
cap = cv2.VideoCapture(vids[1])
token = '6230979992:AAE-f4Cex5FMyAX0QID2aihl_dziVCfwvwQ'
url = f'https://api.telegram.org/bot{token}/sendPhoto'
url_text = f'https://api.telegram.org/bot{token}/sendMessage'

def sendphoto(frame, detections, t, interseccion):
    print("Detecto algo")
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes_io = io.BytesIO(img_encoded.tobytes())

    # Use a separate thread to handle the photo sending
    Thread(target=send_photo_in_thread, args=(img_bytes_io,)).start()

def send_photo_in_thread(img_bytes_io):
    res = requests.post(url, data={'chat_id': chat_id}, files={'photo': img_bytes_io})
    if res.status_code == 200:
        print('La imagen ha sido enviada correctamente.')
    else:
        print('Ocurrió un error al enviar la imagen.')
    print(res)

	
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    detections, t = model.Inference(frame)
    for obj in detections:
       print(obj['class'], obj['conf'], obj['box'])
    print("FPS: {} sec".format(1/t))
    clases_detectadas = [str(obj['class']) for obj in detections]
    # Otro array de strings para la comparación
    clases_detectar = ["arma_corta", "arma_punzocortante", "arma_larga"]
    # Realizar la intersección de los dos arrays de strings
    interseccion = list(set(clases_detectadas) & set(clases_detectar))
    # Imprimir resultados
    #print("Array de strings:", array_de_strings)
    #print("Otro array de strings:", otro_array_de_strings)
    #print("Intersección:", interseccion)
    cv2.imshow("Output", frame)
    if interseccion:
        sendphoto(frame,detections,t,interseccion)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
