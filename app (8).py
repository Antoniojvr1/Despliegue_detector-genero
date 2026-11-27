import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import av
import urllib.request
import os

st.set_page_config(page_title='Detector de Género Ultra-Ligero', layout='centered')
st.title('Detector de Género en Tiempo Real ⚡')
st.write('Versión optimizada de alto rendimiento sin TensorFlow para Streamlit Cloud.')

# Descargar modelos pre-entrenados ligeros de OpenCV para Género si no existen
@st.cache_resource
def load_models():
    # Rostro (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Género (Caffe Model - Solo ~9MB)
    proto_url = 'https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt'
    model_url = 'https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel'
    
    if not os.path.exists('gender_deploy.prototxt'):
        urllib.request.urlretrieve(proto_url, 'gender_deploy.prototxt')
    if not os.path.exists('gender_net.caffemodel'):
        urllib.request.urlretrieve(model_url, 'gender_net.caffemodel')
        
    gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
    return face_cascade, gender_net

try:
    face_cascade, gender_net = load_models()
    MODEL_MEAN_VALUES = (104.0, 117.0, 123.0)
    gender_list = ['Hombre', 'Mujer']
except Exception as e:
    st.error(f'Error al cargar los modelos: {e}')

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format='bgr24')
        img = cv2.flip(img, 1) # Efecto espejo
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            # Extraer e inflar un poco la zona del rostro para el modelo de género
            pad = 15
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad)
            y2 = min(img.shape[0], y + h + pad)
            
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size > 0:
                # Preparar imagen para la Red Neuronal de OpenCV
                blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                
                # Obtener predicción
                gender = gender_list[gender_preds[0].argmax()]
                confidence = gender_preds[0].max() * 100
                
                # Asignar color según el género
                color = (255, 105, 180) if gender == 'Mujer' else (255, 144, 30) # Rosa para mujer, Azul para hombre
                label = f'{gender}: {confidence:.1f}%'
                
                # Dibujar rectángulo y etiqueta
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                text_y = y - 10 if y - 10 > 10 else y + h + 20
                cv2.putText(img, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
        return av.VideoFrame.from_ndarray(img, format='bgr24')

webrtc_streamer(
    key='gender-detection-light',
    video_frame_callback=lambda frame: VideoProcessor().recv(frame),
    media_stream_constraints={'video': True, 'audio': False},
    rtc_configuration={
        'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]
    }
)

st.markdown('---')
st.markdown('### 🎉 ¡Listo para desplegar sin errores!')
st.markdown('1. Sube este nuevo **app.py** a tu repositorio.')
st.markdown('2. Asegúrate de tener el **requirements.txt** ultra-ligero que generamos antes (solo streamlit, av, streamlit-webrtc y opencv-python-headless).')
st.markdown('3. ¡Reinicia tu aplicación en Streamlit Cloud y ahora cargará al instante!')
