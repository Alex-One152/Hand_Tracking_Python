import cv2
import mediapipe as mp
import numpy as np
import time

pTime = 0
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap =cv2.VideoCapture(0)
while True:
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    success, img = cap.read()

    cv2.putText(img, "FPS:" + str(round(fps)), (0, 23), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    # img = np.flip(img, axis=1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(imgRGB)
    #print(res.multi_hand_landmarks)

    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow(" ", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()