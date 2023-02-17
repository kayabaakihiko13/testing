import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration


model_hands=mp.solutions.hands
hands = model_hands.Hands(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def app():
    st.title('camera in stramlit')
    video_frame=st.empty()
    cap=cv2.VideoCapture(0)
    # frame_count = 0
    while True:
        success,image=cap.read()
        if not success:
            break
        result=hands.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        mp_drawing=mp.solutions.drawing_utils
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,hand_landmarks,model_hands.HAND_CONNECTIONS
                )
        video_frame.image(image,channels="BGR")
    cap.release()
    hands.close()    

# Run the app
if __name__ == "__main__":
    app()
