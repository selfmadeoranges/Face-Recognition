import numpy as np
import cv2

xml = 'haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(xml)

cap = cv2.VideoCapture(0) # 노트북 웹캠을 카메라로 사용
cap.set(3,640) # 너비
cap.set(4,480) # 높이

while(True):
    ret, frame = cap.read() 
    frame = cv2.flip(frame, 1) # 좌우 대칭
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.05,5) 
    print("Number of faces detected: " + str(len(faces)))

    if len(faces):
        for (x,y,w,h) in faces:
            face_img = frame[y:y+h, x:x+w] # 인식된 얼굴 이미지 crop
            face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.04, fy=0.04) # 축소
            face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA) # 확대
            frame[y:y+h, x:x+w] = face_img # 인식된 얼굴 영역 모자이크 처리

    cv2.imshow('result', frame)
        
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Esc 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()