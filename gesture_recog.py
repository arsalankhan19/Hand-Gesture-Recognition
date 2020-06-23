import cv2
import os
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras

bgModel = None
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8    # start point/total width
isBgCaptured = 0          # bool, whether the background captured
bgSubThreshold = 50
learningRate = 0
threshold = 60            # binary threshold
blurValue = 41            # GaussianBlur parameter
thresh = 0

gesture_names = {0: 'C',
                 1: 'FIST',
                 2: 'L',
                 3: 'OKAY',
                 4: 'PALM',
                 5: 'PEACE'}

model = tensorflow.keras.models.load_model('my_hand_gesture_model.h5')

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    print(f'pred_array: {pred_array}')
    result = gesture_names[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score

def main():

    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        global isBgCaptured

        bool_value,frame = camera.read() 
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)    
        cv2.imshow("Original", frame)
        
        # Run once background is captured
        if isBgCaptured == 1:
            img = remove_background(frame)
            img = img[0:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
            cv2.imshow('mask', img)

            # convert the image into binary image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)   # cv2.imshow('blur', blur)
            
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        k = cv2.waitKey(10)

        if k == 27:  # press esc to exit all windows at any time
            break


        elif k == ord('b'):  # press 'b' to capture the background
            global bgModel

            if bgModel is None:
                bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

            else:
                bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
           
            isBgCaptured = 1
            print('Background captured')


        elif k == ord('r'):  # press 'r' to reset the background
            time.sleep(1)
            bgModel = None
            isBgCaptured = 0
            print('Reset background')

        elif k == 32:
            # If space bar pressed
            # cv2.imshow('original', frame)
            # copies 1 channel BW image to all 3 RGB channels
            target = np.stack((thresh,) * 3, axis=-1)
            target = cv2.resize(target, (224, 224))
            target = target.reshape(224, 224, 3)
            plt.imshow(target)
            plt.show()
            target = target.reshape(1, 224, 224, 3)
            prediction, score = predict_rgb_image_vgg(target)

            if prediction == 'L':       #LOCK
                command = "xte 'keydown Super_L' 'keydown L' 'keyup Super_L' 'keyup L'"
                os.system(command)

            elif prediction == 'PEACE':  #TERMINAL
                command = "xte 'keydown Control_L' 'keydown Alt_L' 'keydown T' 'keyup Control_L' 'keyup Alt_L' 'keyup T'"
                os.system(command)

            elif prediction == 'OKAY':
                command = "xte 'keydown Control_L' 'keydown Alt_L' 'keydown Page_Down' 'keyup Control_L' 'keyup Alt_L' 'keyup Page_Down'"
                os.system(command)

            elif prediction == 'FIST':
                command = "xte 'keydown Control_L' 'keydown Alt_L' 'keydown Page_Up' 'keyup Page_Up'"
                os.system(command)



    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
