from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from PIL import Image
import tensorflow.compat.v1 as tf
video= 0
modeldir = './model/20170511-185253.pb' #20180408-102900.pb, 20180402-114759.pb, 20170511-185253.pb
classifier_filename = './class/170511_50_2_cf.pkl' #classifier(30_180402).pkl, classifier(30_180408).pkl, classifier(30_170511).pkl
npy='./npy'
train_img="./train_img"

total_number = 0
best_class_number = 0
str_class_number = 0

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        minsize = 20  # minimum size of face
        threshold = [0.6,0.7,0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        batch_size =50      #학습시킬 데이터 양에 따라 변경시 효과적
        image_size = 182
        input_image_size = 160
        HumanNames = os.listdir(train_img)
        HumanNames.sort()
        print('Loading Model')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile,encoding='latin1')
        
        video_capture = cv2.VideoCapture(video)
        print('Start Recognition')
        while True:
            ret, frame = video_capture.read()
            frame = cv2.flip(frame, 1) #w좌우 반전
            #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
            timer =time.time()
            
            cv2.imshow('low', frame)

            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)
            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            faceNum = bounding_boxes.shape[0]
            if faceNum > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(frame.shape)[0:2]
                cropped = []
                scaled = []
                scaled_reshape = []
                for i in range(faceNum):
                    emb_array = np.zeros((1, embedding_size))
                    xmin = int(det[i][0])
                    ymin = int(det[i][1])
                    xmax = int(det[i][2])
                    ymax = int(det[i][3])
                    try:
                        # inner exception
                        if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                            print('Face is very close!')
                            continue
                        cropped.append(frame[ymin:ymax, xmin:xmax,:])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                        total_number+=1

                        if best_class_probabilities>0.8:

                            best_class_number+=1

                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)    #boxing face
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]    #인식된 얼굴 이름
                                    accuracy = str(int(best_class_probabilities[0] * 100)) + '%'    #정확도
                                    print("Predictions : [ name: {} , accuracy: {:.3f} ]".format(result_names, best_class_probabilities[0]))
                                    cv2.rectangle(frame, (xmin, ymin-45), (xmax, ymin-2), (255, 255,255), -1)   #이름, 정확도 쓰기위한 바탕
                                    cv2.putText(frame, result_names, (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=1)  #이름
                                    cv2.putText(frame, accuracy, (xmin,ymin-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=1)     #정확도
                                    
                        else :

                            str_class_number+=1

                            print(ymin,ymax,xmin,xmax)
                            face_img = frame[ymin:ymax, xmin:xmax] # 인식된 얼굴 이미지 crop
                            face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.04, fy=0.04) # 축소
                            face_img = cv2.resize(face_img, (xmax-xmin, ymax-ymin), interpolation=cv2.INTER_AREA) # 확대
                            frame[ymin:ymax, xmin:xmax] = face_img # 인식된 얼굴 영역 모자이크 처리
                    
                            # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)  #boxing face
                            # cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (255, 255,255), -1)   #이름 쓰기위한 바탕
                            # cv2.putText(frame, "stranger", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=1)    #stranger 표기
                    except:   
                        print("error")
                       
            # endtimer = time.time()
            #fps = 1/(endtimer-timer)
            #cv2.rectangle(frame,(15,30),(135,60),(0,255,255),-1)
            #cv2.putText(frame, "fps: {:.2f}".format(fps), (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imshow('Real-time Face Recognition and mosaic', frame)
            key= cv2.waitKey(1)
            if key== 113: # "q"
                print('total number : {}, best_class_number : {}, str_class_number : {}'.format(total_number, best_class_number, str_class_number))
                break
        video_capture.release()
        cv2.destroyAllWindows()