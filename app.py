import cv2 as c
import numpy as np
import mediapipe as m
import joblib as j
sign = j.load(open('HANDS', 'rb'))
h_sgn = sign['model']
cp = c.VideoCapture(0)
m_hn = m.solutions.hands
m_draw = m.solutions.drawing_utils
m_draw_styl = m.solutions.drawing_styles
hnds = m_hn.Hands(static_image_mode=True, min_detection_confidence=0.3)

target_dic = {0: 'ABUSIVE', 1: 'HELLO', 2: 'LITTLE', 3: 'OK', 4: 'VICTORY'}

while True:
    dt_au = []
    x_ = []
    y_ = []
    ret, frame = cp.read()
    H, W,_= frame.shape

    im_rgb = c.cvtColor(frame, c.COLOR_BGR2RGB)

    result = hnds.process(im_rgb)
    if result.multi_hand_landmarks:
        for hnd_lndmrk in result.multi_hand_landmarks:
            m_draw.draw_landmarks(frame, hnd_lndmrk, m_hn.HAND_CONNECTIONS,
                                  landmark_drawing_spec=m_draw_styl.get_default_hand_landmarks_style(),
                                  connection_drawing_spec=m_draw_styl.get_default_hand_connections_style())
        for hnd_lndmrk in result.multi_hand_landmarks:
            for i in range(len(hnd_lndmrk.landmark)):
                x = hnd_lndmrk.landmark[i].x
                y = hnd_lndmrk.landmark[i].y
                dt_au.append(x)
                dt_au.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)

        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        # Make a prediction using the loaded model
        pred = h_sgn.predict([np.asarray(dt_au)])
        pre_sign = target_dic.get(int(pred[0]))

     

        c.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        c.putText(frame, pre_sign, (x1, y1), c.FONT_HERSHEY_COMPLEX, 1.3, (0,0, 255), 3, c.LINE_AA)

    c.imshow("frame", frame)
    if c.waitKey(25) & 0xff == ord('q'):
        break

cp.release()
c.destroyAllWindows()
