import sys
import cv2
import imutils
from yoloDet import YoloTRT
import requests
import tempfile
import io
from threading import Thread
import time

# use path for library and engine file
engines = [ 'yolov7/build/best.engine', 'yolov7/build/bestYOLOv7Prueba2.engine']
model = YoloTRT(library="yolov7/build/libmyplugins.so", engine=engines[1], conf=0.6, yolo_ver="v7")
chat_id = '1045458604'
vids = [ 'videos/testvideo.mp4', 'video2.mp4' , 'armalarga1.mp4', 'armalarga2.mp4']
cap = cv2.VideoCapture(vids[1])
token = '6230979992:AAE-f4Cex5FMyAX0QID2aihl_dziVCfwvwQ'
url = f'https://api.telegram.org/bot{token}/sendPhoto'
url_text = f'https://api.telegram.org/bot{token}/sendMessage'
archivo_time = "tiempos.txt"
inicio_total = time.time();
contador_frames = 0;
hilos_join = [];
def sendphoto(frame, detections, t, interseccion,tiempo_before_detection,tiempo_after_detection):
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes_io = io.BytesIO(img_encoded.tobytes())

    # Use a separate thread to handle the photo sending
    hilo_ = Thread(target=send_photo_in_thread, args=(img_bytes_io,tiempo_before_detection,tiempo_after_detection,));
    hilo_.start();
    hilos_join.append(hilo_)

def send_photo_in_thread(img_bytes_io,tiempo_before_detection,tiempo_after_detection):
    res = requests.post(url, data={'chat_id': chat_id}, files={'photo': img_bytes_io})
    #if res.status_code == 200:
    #    print('La imagen ha sido enviada correctamente.')
    #else:
    #    print('Ocurrió un error al enviar la imagen.')
    tiempo_after_sent = time.time();
    tiempo_deteccion = tiempo_after_detection - tiempo_before_detection
    tiempo_envio = tiempo_after_sent - tiempo_after_detection
    tiempo_proceso = tiempo_after_sent - tiempo_before_detection
    #f.write("Tiempo Deteccion de la Red Neuronal: " + str(tiempo_deteccion) + " \nTiempo Envio desde Deteccion de la Red Neuronal: " + str(tiempo_envio) + "\nTiempo Proceso Frame: " + str(tiempo_proceso))
    #f.write("t: {}".format(t))
    with open(archivo_time,'a') as f:
        f.write("\n ------------------------------  ")
        f.write(" \n Tiempo Proceso Frame: {} ".format(tiempo_proceso) + " \n Tiempo Deteccion de la Red Neuronal: {}".format(tiempo_deteccion) + " \n Tiempo Envio desde Deteccion de la Red Neuronal: {}".format(tiempo_envio))
    #print(f"Tiempo Deteccion de la Red Neuronal: {tiempo_deteccion}  \nTiempo Envio desde Deteccion de la Red Neuronal: {tiempo_envio} \nTiempo Proceso Frame: {tiempo_proceso}") 

with open(archivo_time,'w') as f:
    f.write("Iniciando Proceso\n")	
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    if contador_frames % 5 == 0:
        tiempo_before_detection = time.time()
        detections, t = model.Inference(frame)
        tiempo_after_detection = time.time()
        #for obj in detections:
        #   print(obj['class'], obj['conf'], obj['box'])
        #print("FPS: {} sec".format(1/t))
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
    if contador_frames % 5 == 0:
        if interseccion:
            sendphoto(frame,detections,t,interseccion,tiempo_before_detection,tiempo_after_detection)
    contador_frames += 1;
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
for hilo_ in hilos_join:
    hilo_.join();
fin_total = time.time();
tiempo_total = fin_total - inicio_total;
with open(archivo_time,'a') as f:
    f.write("\n \n \n Tiempo Total de Ejecucion: {}".format(tiempo_total))
    f.write(" \nTotal de Frames: {}".format(contador_frames))

