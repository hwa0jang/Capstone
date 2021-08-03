import numpy as np
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image
from sklearn.svm import SVC
from faceidentify.SVMclassifier import model as svm
from faceidentify.SVMclassifier import out_encoder

#### FOR GAZE AND MOTION ####
import time
import argparse
import cv2
import os.path as osp
from detectheadposition import headpose
from gaze_tracking import GazeTracking

#### FOR WARNING ####
import pygame  # For play Sound
import time  # For sleep
import threading  # For multi thread
# from tkinter import *
# import tkinter.messagebox

# def Msgbox1():
#     tkinter.messagebox.showwarning("경고", "집중하세요")

# Warning Sound


def Sound():
    pygame.mixer.init()
    music = pygame.mixer.Sound("Warning/warning.wav")
    music.play()
    time.sleep(5)

# Settle Cheater
def Fail(timee, redcard):
    if redcard >= timee/3:
        print("===부정행위자 입니다===")

# get the face embedding for one face


def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # print(face_pixels.shape)
    # transform face into one sample
    # expand dims adds a new dimension to the tensor
    samples = np.expand_dims(face_pixels, axis=0)
    # print(samples.shape)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

# Print Result


def PrintResult(x, y):
    print("###############--RESULT--#################")
    print("yellocard:", x, "/ redcard", y)
    print("###########################################")

# point can't get negative


def notnegative(x):
    if x < 0:
        return 0
    else:
        return x



# main function
def main(args):
    filename = args["input_file"]
#<<<<<<< HEAD
    faceCascade = cv2.CascadeClassifier('C:/Capstone/models/haarcascade_frontalface_default.xml')
    model = load_model('C:/Capstone/models/facenet_keras.h5')
#=======
    faceCascade = cv2.CascadeClassifier(
        'models/haarcascade_frontalface_default.xml')
    model = load_model('models/facenet_keras.h5')
#>>>>>>> 757e3559f2acad057224670a45fca1fc2d17309e

    if filename is None:
        isVideo = False
        #url='http://192.168.0.06:8091/?action=stream'
        #webcam = cv2.VideoCapture(url)
        webcam = cv2.VideoCapture(0)
        webcam.set(3, args['wh'][0])
        webcam.set(4, args['wh'][1])       
    else:
        isVideo = True
        webcam = cv2.VideoCapture(filename)
        fps = webcam.get(cv2.webcam_PROP_FPS)
        width = int(webcam.get(cv2.webcam_PROP_FRAME_WIDTH))
        height = int(webcam.get(cv2.webcam_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        name, ext = osp.splitext(filename)
        out = cv2.VideoWriter(args["output_file"],
                              fourcc, fps, (width, height))
        


    UserName = (input("시험응시자의 이름을 입력하시오.: "))
    checktime = 1
    start_check= time.time() + (10 * checktime)
    checktime_end = time.time() + (60 * checktime)
    while (webcam.isOpened()):# Infinity Loop for Detect Cheating for Online test
        ret, frame = webcam.read()  # Read wabcam
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)  # face structure
        for (x, y, w, h) in faces:
            # take the face pixels from the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # turn the face pixels back into an image
            crop_frame = frame[y:y+h, x:x+w]
            # resize the image to meet the size requirment of facenet
            new_crop = Image.fromarray(crop_frame)
            # turn the image back into a tensor
            new_crop = new_crop.resize((160, 160))
            # get the face embedding using the face net model
            crop_frame = np.asarray(new_crop)
            # it is a 1d array need to reshape it as a 2d tensor for svm
            face_embed = get_embedding(model, crop_frame)
            # predict using our SVM model
            face_embed = face_embed.reshape(-1, face_embed.shape[0])
            pred = svm.predict(face_embed)  # get the prediction probabiltiy
            # pred_prob has probabilities of each class
            pred_prob = svm.predict_proba(face_embed)

            # get name
            class_index = pred[0]
            class_probability = pred_prob[0, class_index] * 100
            predict_names = out_encoder.inverse_transform(pred)
            text = '%s (%.3f%%)' % (predict_names[0], class_probability)
            cv2.putText(frame, text, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('capstone', frame)
        cv2.waitKey(1)   
        if(time.time() > start_check and predict_names[0]==UserName and class_probability > 80):
            print("얼굴이 일치합니다. 시험을 시작하겠습니다.")
            break 
        if time.time() > checktime_end:
           print("얼굴이 일치하지 않아 시험에 응시하지 못합니다.")
           quit()
           break 
       
    # Variable Setting
    hpd = headpose.HeadposeDetection(
        args["landmark_type"], args["landmark_predictor"])  # import headpose
    gaze = GazeTracking()  # import gazetracking
    yellocard = 0
    redcard = 0
    tempval = 0
    # Input time for limit test time
    timee = int(input("시험 시간을 입력하세요(Minute): "))
    max_time_end = time.time() + (60 * timee)


    check_angle=time.time()+(10*checktime)
    while(webcam.isOpened()):# Infinity Loop for Detect Cheating for Online test
        ret, frame = webcam.read()  # Read wabcam
        gaze.refresh(frame)
        frame = gaze.annotated_frame()  # Mark pupil for frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)  # face structure

        # Get point from pupil
        if gaze.is_blinking():
            yellocard = yellocard - 1
            yellocard = notnegative(yellocard)
        elif gaze.is_right():
            yellocard = yellocard - 1
            yellocard = notnegative(yellocard)
        elif gaze.is_left():
            yellocard = yellocard - 1
            yellocard = notnegative(yellocard)
        elif gaze.is_center():
            yellocard = yellocard - 1
            yellocard = notnegative(yellocard)
        else:
            yellocard = yellocard + 2

        # Get redcard optiom
        if yellocard > 50:
            yellocard = 0
            tempval = tempval + 1
            redcard = redcard + 1

        # if get 1redcard, then give Aural and Text Warning(Loop)
        if tempval == 1:
            text1 = "WARNING"
            cv2.putText(frame, text1, (10, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 255), 2)
            my_thread = threading.Thread(target=Sound)
            my_thread.start()
            tempval = 0

    # if you are not GPU environment, Do not run this code by # --------------
        # if get 2redcard, then give Picture Warning(Once)
        if redcard == 2:
            warn_img = cv2.imread("Warning/warning.png", cv2.IMREAD_COLOR)
            cv2.imshow('Warning', warn_img)
            cv2.waitKey(1)
            redcard = 2.1
    # -----------------------------s------------------------------------------
        # Get log consistently
        print("<< *의심수준:", yellocard, " || ", "*경고횟수:", redcard, " >>")
        #cv2.destroyWindow('Warning')
        # Detect head position
        if isVideo:
            frame, angles = hpd.process_image(frame)
            if frame is None:
                break
            else:
                out.write(frame)
        else:
            frame, angles = hpd.process_image(frame)
            if angles is None:
                print("경고! 응시자가 사라졌습니다")
                if time.time() > check_angle:
                    redcard= timee/3+ redcard
                    print("지속적으로 카메라 앵글 밖으로 나갔으므로, 시험을 강제종료합니다.")
                    PrintResult(yellocard, redcard)
                    Fail(timee, redcard)
                    quit()
                else:
                    pass
                
            else:  # angles = [x,y,z] , get point from headposition
                if angles[0] > 15 or angles[0] < -15 or angles[1] > 15 or angles[1] < -15 or angles[2] > 15 or angles[2] < -15:
                    yellocard = yellocard + 2
                else:
                    yellocard = yellocard - 1
                    yellocard = notnegative(yellocard)

        yellocard = yellocard + hpd.yello(frame)
        if yellocard < 0:
            yellocard = notnegative(yellocard)

        # Display the resulting frame
            cv2.imshow('capstone', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("관리자에 의해 시험이 강제 종료 되었습니다")
            PrintResult(yellocard, redcard)
            Fail(timee, redcard)
            break
        elif time.time() > max_time_end:
            print(timee, "분의 시험이 종료되었습니다.")
            PrintResult(yellocard, redcard)
            Fail(timee, redcard)
            break

    # When everything done, release the webcam
    webcam.release()
    cv2.destroyAllWindows()
    quit()
    if isVideo:
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', metavar='FILE', dest='input_file', default=None,
                        help='Input video. If not given, web camera will be used.')
    parser.add_argument('-o', metavar='FILE', dest='output_file',
                        default=None, help='Output video.')
    parser.add_argument('-wh', metavar='N', dest='wh',
                        default=[720, 480], nargs=2, help='Frame size.')
    parser.add_argument('-lt', metavar='N', dest='landmark_type',
                        type=int, default=1, help='Landmark type.')
    parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor',
                        default='gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
    args = vars(parser.parse_args())
#<<<<<<< HEAD
    main(args)
#=======
    main(args)
#>>>>>>> 757e3559f2acad057224670a45fca1fc2d17309e
