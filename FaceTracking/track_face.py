import find_face
import cv2
import numpy as np
dlib_obj = find_face.FindFace()
camera = cv2.VideoCapture(0)
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
ret, frame = camera.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
mask = np.zeros_like(frame)
while True:
    ret, frame = camera.read()
    blurred_frame = cv2.medianBlur(frame, 5)

    cv2.imshow("Blur Image", blurred_frame[::2,::2,:])



    if ret:
        boxes = dlib_obj.getfaces(frame[::2,::2,:])
        boxes = np.asarray(boxes) * 2
        center = None
        for box in boxes:
            frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,230),thickness=2)
            center = np.int32([box[0]+((box[2]-box[0])//2),box[1]+((box[3]-box[1])//2)])
            frame = cv2.circle(frame, (center[0],center[1]), 5, (255,255,255), thickness=2)

        if center is not None:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            c = np.asarray([[center]])
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0 , None, **lk_params)
            old_gray = frame_gray.copy()
            p0=p1
            print("=====")
            print("length",len(p1))
            print(p1)
            print("=====")
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), (255, 255, 255), 2)
                frame = cv2.circle(frame, (a, b), 5, (255, 2, 255), -1)
            frame = cv2.add(frame, mask)
    cv2.imshow("Face Tracker", frame[::2,::2,:])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()