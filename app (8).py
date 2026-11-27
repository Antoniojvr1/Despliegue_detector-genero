import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
from deepface import DeepFace
import av

st.set_page_config(page_title='Detector de Genero con IA', layout='centered')
st.title('Detector de Genero en Tiempo Real')
st.write('Esta aplicacion utiliza tu camara web, Haar Cascades para localizar rostros y DeepFace para predecir el genero.')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class VideoProcessor:
    def __init__(self):
        self.frame_count = 0
        self.tracked_faces = []

    def recv(self, frame):
        img = frame.to_ndarray(format='bgr24')
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(50, 50))
        new_tracked_faces = []

        for (x, y, w, h) in faces:
            cx, cy = x + w // 2, y + h // 2
            matched_label = 'Analizando...'
            matched_color = (0, 255, 0)

            for prev in self.tracked_faces:
                if abs(cx - prev['cx']) < 50 and abs(cy - prev['cy']) < 50:
                    matched_label = prev['label']
                    matched_color = prev['color']
                    break

            if self.frame_count % 10 == 0 or matched_label == 'Analizando...':
                try:
                    face_crop = img[y:y+h, x:x+w]
                    if face_crop.size > 0:
                        analysis = DeepFace.analyze(
                            img_path=face_crop,
                            actions=['gender'],
                            detector_backend='skip',
                            enforce_detection=False,
                            silent=True
                        )
                        res = analysis[0] if isinstance(analysis, list) else analysis
                        w_score = res['gender'].get('Woman', 0.0)
                        m_score = res['gender'].get('Man', 0.0)

                        if w_score > 40.0:
                            matched_label = f'Mujer: {w_score:.1f}%'
                            matched_color = (180, 105, 255)
                        else:
                            matched_label = f'Hombre: {m_score:.1f}%'
                            matched_color = (255, 144, 30)
                except Exception as e:
                    pass

            new_tracked_faces.append({'cx': cx, 'cy': cy, 'label': matched_label, 'color': matched_color})
            cv2.rectangle(img, (x, y), (x+w, y+h), matched_color, 2)
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(img, matched_label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, matched_color, 2)

        self.tracked_faces = new_tracked_faces
        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format='bgr24')

webrtc_streamer(
    key='gender-detection',
    video_frame_callback=lambda frame: VideoProcessor().recv(frame),
    media_stream_constraints={'video': True, 'audio': False},
    rtc_configuration={
        'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]
    }
)

st.markdown('---')
st.markdown('### Instrucciones para Desplegar:')
st.markdown('1. Sube estos tres archivos (app.py, requirements.txt, packages.txt) a tu repositorio.')
st.markdown('2. Conecta tu GitHub con Streamlit Community Cloud.')
st.markdown('3. ¡Disfruta de tu app desplegada!')