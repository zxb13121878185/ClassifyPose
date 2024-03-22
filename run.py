# Pose Detections with Model
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import mediapipe as mp 
import pickle

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def display_classify_pose(cap, model):
    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers.
    mp_holistic = mp.solutions.holistic     # Mediapipe Solutions.

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False    

                # Make Detections
                results = holistic.process(image)

                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    # Concate rows
                    row = pose_row

                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    # print(f'class: {body_language_class}, prob: {body_language_prob}')
                    # 绘制跟随人体的标签
                    # coords = tuple(np.multiply(
                    #     np.array(
                    #         (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                    #         results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                    #     [640,480]
                    # ).astype(int))

                    # cv2.rectangle(image, 
                    #             (coords[0], coords[1]+5), 
                    #             (coords[0]+len(body_language_class)*20, coords[1]-30), 
                    #             (245, 117, 16), -1)

                    # cv2.putText(image, body_language_class, coords, 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Get status box
                    cv2.rectangle(image, (0,0), (180, 110), (0, 0, 150), -1)
                    image=cv2AddChineseText(image, '类别', (15,10),(255, 255, 255), 30)
                    image=cv2AddChineseText(image, body_language_class,  (15,70),(0, 255, 0), 30)
                    image=cv2AddChineseText(image, '评分', (90,10),(255, 255, 255), 30)
                    body_language_prob = body_language_prob*100
                    image=cv2AddChineseText(image,
                        str(round(body_language_prob[np.argmax(body_language_prob)],2)),  
                        (90,70),(0, 255, 0), 30)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                except:
                    pass

                cv2.imshow('Raw Webcam Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            else:
                break

    print('Done!')
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    model_weights = './model_weights/weights_body_language.pkl'  
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920) #设置窗口大小
    cap.set(4, 1080) 
    # Load Model.
    with open(model_weights, 'rb') as f:
        model = pickle.load(f)
    
    display_classify_pose(cap=cap, model=model)
