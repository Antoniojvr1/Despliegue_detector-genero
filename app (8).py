import streamlit as st
import cv2
import numpy as np
import urllib.request
import os

st.set_page_config(page_title='Detector de Género Estable', layout='centered')
st.title('Detector de Género con Cámara Web 📸')
st.write('Versión estable de máxima compatibilidad para Streamlit Cloud.')

# Descargar modelos pre-entrenados ligeros de OpenCV para Género
@st.cache_resource
def load_models():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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

# Widget de cámara nativo de Streamlit
img_file = st.camera_input('¡Tómate una foto para analizar tu género!')

if img_file is not None:
    # Convertir la imagen capturada para OpenCV
    bytes_data = img_file.getvalue()
    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Procesar imagen
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60))
    
    if len(faces) == 0:
        st.warning('No se detectó ningún rostro. ¡Inténtalo de nuevo acercándote un poco más!')
    else:
        for (x, y, w, h) in faces:
            pad = 15
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(cv_img.shape[1], x + w + pad)
            y2 = min(cv_img.shape[0], y + h + pad)
            
            face_crop = cv_img[y1:y2, x1:x2]
            if face_crop.size > 0:
                # Predicción del género con la red ligera
                blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                
                gender = gender_list[gender_preds[0].argmax()]
                confidence = gender_preds[0].max() * 100
                
                # Dibujar en la imagen
                color = (180, 105, 255) if gender == 'Mujer' else (30, 144, 255) # RGB adaptable
                cv2.rectangle(cv_img, (x, y), (x+w, y+h), color, 4)
                label = f'{gender} ({confidence:.1f}%)'
                cv2.putText(cv_img, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Mostrar el resultado final procesado
        st.image(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB), caption='Resultado del Análisis', use_container_width=True)
        st.success('¡Análisis completado!')
