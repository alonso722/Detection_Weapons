import sys
import cv2
import imutils
from yoloDet import YoloTRT
import requests
import tempfile
import io
from threading import Thread
import time
import datetime

# use path for library and engine file
engines = [ 'yolov7/build/best.engine', 'yolov7/build/bestYOLOv7Prueba2.engine']
model = YoloTRT(library="yolov7/build/libmyplugins.so", engine=engines[1], conf=0.7, yolo_ver="v7")
chat_id = '1045458604'		#ARTURO
#chat_id = '6841694914'		#USIEL
vids = [ 'videos/arma_corta/video_arma_corta_8.mp4', 'videos/arma_larga/video_arma_larga_3.mp4' , 'videos/arma_punzocortante/.mp4', 'videos/sin_arma/.mp4']
video_open = vids[0]; #Video a Ejecutar
cap = cv2.VideoCapture(video_open)
token = '6230979992:AAE-f4Cex5FMyAX0QID2aihl_dziVCfwvwQ'
url = f'https://api.telegram.org/bot{token}/sendPhoto'
url_text = f'https://api.telegram.org/bot{token}/sendMessage'
inicio_total = time.time();
contador_frames = 0;
frames_a_detectar = 0;
fecha_actual = datetime.datetime.now()
archivo_time = f"logs/{video_open}_{fecha_actual}.txt"
hilos_join = [];
divisor_frames = 5; #Modificar
def sendphoto(frame, detections, t, interseccion,tiempo_before_detection,tiempo_after_detection):
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes_io = io.BytesIO(img_encoded.tobytes())

    # Use a separate thread to handle the photo sending
    hilo_ = Thread(target=send_photo_in_thread, args=(img_bytes_io,tiempo_before_detection,tiempo_after_detection,interseccion,));
    hilo_.start();
    hilos_join.append(hilo_)

def send_photo_in_thread(img_bytes_io,tiempo_before_detection,tiempo_after_detection,interseccion):
    #caption = "\nSe ha detectado la(s) clase(s): ";
    caption = "\n";
    for clase in interseccion:
        caption = caption + str(clase["class"]) + " - " + str(clase["conf"]) + " , "
    tiempo_after_sent = time.time();
    tiempo_deteccion = tiempo_after_detection - tiempo_before_detection
    tiempo_envio = tiempo_after_sent - tiempo_after_detection
    tiempo_proceso = tiempo_after_sent - tiempo_before_detection
    #caption = caption + "\nTiempo Deteccion de la Red Neuronal: " + str(tiempo_deteccion) + " \nTiempo Envio desde Deteccion de la Red Neuronal: " + str(tiempo_envio) + "\nTiempo Proceso Frame: " + str(tiempo_proceso);
    caption += f" | {tiempo_deteccion:.2f},{tiempo_envio:.2f},{tiempo_proceso:.2f}"
    res = requests.post(url, data={'chat_id': chat_id, 'caption': caption}, files={'photo': img_bytes_io})
    #if res.status_code == 200:
    #    print('La imagen ha sido enviada correctamente.')
    #else:
    #    print('Ocurrió un error al enviar la imagen.')
    with open(archivo_time,'a') as f:
        f.write("\n ------------------------------  ")
        #f.write(" \n Tiempo Proceso Frame: {} ".format(tiempo_proceso) + " \n Tiempo Deteccion de la Red Neuronal: {}".format(tiempo_deteccion) + " \n Tiempo Envio desde Deteccion de la Red Neuronal: {}".format(tiempo_envio) + " \n" + caption)
        f.write(caption)

with open(archivo_time,'w') as f:
    f.write("Iniciando Proceso\n")	
while True:
    ret, frame = cap.read()
    try:
        frame = imutils.resize(frame, width=600)
        if contador_frames % divisor_frames == 0:
            tiempo_before_detection = time.time()
            detections, t = model.Inference(frame)
            tiempo_after_detection = time.time()
            clases_detectar = ["arma_corta", "arma_punzocortante", "arma_larga", "posicion_arma_corta", "posicion_arma_punzocortante", "posicion_arma_larga"]
            interseccion = [obj for obj in detections if obj["class"] in clases_detectar]
            frames_a_detectar += 1;
        cv2.imshow("Video", frame)
        if contador_frames % divisor_frames == 0:
            if interseccion:
                sendphoto(frame,detections,t,interseccion,tiempo_before_detection,tiempo_after_detection)
        contador_frames += 1;
        key = cv2.waitKey(1)
        if key == ord('q'):
            break;
    except Exception as e:
        print("Termino")
        break;
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
    f.write("\n \n \nTiempo Total de Ejecucion: {}".format(tiempo_total))
    f.write(" \nTotal de Frames Seleccionados: {}".format(frames_a_detectar))
    f.write(" \nTotal de Frames: {}".format(contador_frames))
print("\nCreando Archivo Log... ");
url_document = f'https://api.telegram.org/bot{token}/sendDocument'
with open(archivo_time, 'rb') as archivo:
    params = {
        'chat_id': chat_id,
    }
    files = {'document': archivo}
    response_document = requests.post(url_document, params=params, files=files)
    if response_document.status_code == 200:
        print('El archivo ha sido enviado con éxito.')
    else:
        print('Hubo un error al enviar el archivo:', response.text)
print("Finalizando Programa...")
