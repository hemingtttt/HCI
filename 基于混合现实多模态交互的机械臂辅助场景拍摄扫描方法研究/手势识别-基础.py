import cv2
import mediapipe as mp
import math
"""
OpenCV：打开摄像头，读取视频帧，图像颜色转换，在画面上画框、写字、显示窗口

MediaPipe：手部关键点检测

math：数学库，计算欧几里得距离
"""



mp_hands = mp.solutions.hands #初始化手部模块
mp_drawing = mp.solutions.drawing_utils #初始化绘画工具


def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y) #计算距离


def get_finger_states(hand_landmarks, handedness_label):
    """
    返回五根手指是否伸直:
    [thumb, index, middle, ring, pinky]
    1 = 伸直, 0 = 弯曲
    """
    lm = hand_landmarks.landmark #手指21个关键点列表
    fingers = [] #初始化手指状态列表

    # 1) 拇指判断
    # 用x坐标粗略判断，左右手方向相反
    if handedness_label == "Right":
        thumb_open = 1 if lm[4].x < lm[3].x else 0
    else:
        thumb_open = 1 if lm[4].x > lm[3].x else 0
    fingers.append(thumb_open)   #thumb

    # 2) 其他四指：tip 在 pip 上方（图像坐标 y 越小越靠上）
    fingers.append(1 if lm[8].y < lm[6].y else 0)    # index
    fingers.append(1 if lm[12].y < lm[10].y else 0)  # middle
    fingers.append(1 if lm[16].y < lm[14].y else 0)  # ring
    fingers.append(1 if lm[20].y < lm[18].y else 0)  # pinky

    return fingers


def recognize_gesture(hand_landmarks, handedness_label):
    lm = hand_landmarks.landmark
    fingers = get_finger_states(hand_landmarks, handedness_label)

    thumb, index, middle, ring, pinky = fingers

    # OK 手势：食指指尖和拇指指尖距离很近，其他手指张开
    thumb_index_dist = distance(lm[4], lm[8])
    ok_gesture = thumb_index_dist < 0.05 and middle == 1 and ring == 1 and pinky == 1

    if ok_gesture:
        return "OK"

    total_open = sum(fingers)

    if total_open == 0:
        return "FIST"

    if total_open == 5:
        return "PALM"

    if fingers == [0, 1, 0, 0, 0]:
        return "ONE"

    if fingers == [0, 1, 1, 0, 0]:
        return "TWO"

    if fingers == [0, 1, 1, 1, 0]:
        return "THREE"

    if fingers == [0, 1, 1, 1, 1]:
        return "FOUR"

    if fingers == [1, 1, 1, 1, 1]:
        return "FIVE"

    return "UNKNOWN"


def main():
    cap = cv2.VideoCapture(0) #打开摄像头

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            is_open, frame = cap.read() #读取一帧画面
            if not is_open:
                print("无法读取摄像头")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #OpenCV图像格式是BGR，MediaPipe图像格式是RGB
            results = hands.process(rgb)

            #检测手的标签和姿势
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    label = handedness.classification[0].label  # Left / Right
                    gesture = recognize_gesture(hand_landmarks, label)

                    # 画关键点
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # 取手部包围框大致位置用于显示文字
                    h, w, _ = frame.shape
                    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label}: {gesture}",
                        (x_min, y_min - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

            cv2.putText(
                frame,
                "Press Q to quit",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

            cv2.imshow("Hand Gesture Recognition", frame) #显示窗口

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release() #释放摄像头
    cv2.destroyAllWindows() #关闭窗口


if __name__ == "__main__":
    main()