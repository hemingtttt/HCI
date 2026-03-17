import cv2
import mediapipe as mp
import math
"""
OpenCV：打开摄像头，读取视频帧，图像颜色转换，在画面上画框、写字、显示窗口

MediaPipe：手部关键点检测

math：数学库，计算欧几里得距离
"""


mp_holistic = mp.solutions.holistic #初始化手部模块
mp_drawing = mp.solutions.drawing_utils #初始化绘画工具


def clamp(v, lo, hi):
    return max(lo, min(hi, v)) #clip函数


def to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h) #坐标转化


def dist2d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y) #计算距离


def angle_deg(dx, dy):
    return math.degrees(math.atan2(-dy, dx)) #计算角度


#方向转化
def direction_from_vector(dx, dy, deadzone=0.02):
    """
    把二维向量转成方向文字
    """
    if abs(dx) < deadzone and abs(dy) < deadzone:
        return "STILL"

    ang = angle_deg(dx, dy)

    # 8方向
    if -22.5 <= ang < 22.5:
        return "RIGHT"
    elif 22.5 <= ang < 67.5:
        return "UP-RIGHT"
    elif 67.5 <= ang < 112.5:
        return "UP"
    elif 112.5 <= ang < 157.5:
        return "UP-LEFT"
    elif ang >= 157.5 or ang < -157.5:
        return "LEFT"
    elif -157.5 <= ang < -112.5:
        return "DOWN-LEFT"
    elif -112.5 <= ang < -67.5:
        return "DOWN"
    else:
        return "DOWN-RIGHT"

# 判断可信度
def avg_visibility(landmarks, ids):
    vals = []
    for i in ids:
        lm = landmarks[i]
        if hasattr(lm, "visibility"):
            vals.append(lm.visibility)
    return sum(vals) / len(vals) if vals else 1.0


# 身体朝向判断（启发式）
def get_body_orientation(pose_landmarks):
    """
    基于肩宽、左右肩相对位置、鼻子相对肩中心位置做启发式判断：
    - FRONT
    - LEFT_TURN
    - RIGHT_TURN
    - BACK
    - UNKNOWN
    """
    lm = pose_landmarks.landmark

    L_SHOULDER = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
    R_SHOULDER = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value
    NOSE = mp_holistic.PoseLandmark.NOSE.value
    L_HIP = mp_holistic.PoseLandmark.LEFT_HIP.value
    R_HIP = mp_holistic.PoseLandmark.RIGHT_HIP.value

    needed = [L_SHOULDER, R_SHOULDER, NOSE, L_HIP, R_HIP]
    if avg_visibility(lm, needed) < 0.45:
        return "UNKNOWN"

    ls = lm[L_SHOULDER]
    rs = lm[R_SHOULDER]
    nose = lm[NOSE]
    lh = lm[L_HIP]
    rh = lm[R_HIP]

    shoulder_width = abs(ls.x - rs.x)
    hip_width = abs(lh.x - rh.x)
    torso_width = (shoulder_width + hip_width) / 2.0

    shoulder_center_x = (ls.x + rs.x) / 2.0
    nose_offset = nose.x - shoulder_center_x

    # 宽度很窄：更接近侧身或背身
    if torso_width < 0.10:
        # 鼻子还明显可见，按侧身判断
        if abs(nose_offset) > 0.03:
            return "LEFT_TURN" if nose_offset < 0 else "RIGHT_TURN"
        else:
            return "SIDE/BACK"

    # 正面时鼻子通常接近肩中心；转身时偏向一侧
    if nose_offset < -0.05:
        return "LEFT_TURN"
    elif nose_offset > 0.05:
        return "RIGHT_TURN"

    # 当肩和胯都比较展开，同时鼻子居中，更像正面
    if torso_width >= 0.16:
        return "FRONT"

    return "UNKNOWN"


# 是否抬手
def is_hand_raised(pose_landmarks, side="left"):
    """
    用 wrist 是否高于 shoulder / mouth 一带来判断抬手
    返回: (bool, text)
    """
    lm = pose_landmarks.landmark
    PL = mp_holistic.PoseLandmark

    if side == "left":
        wrist = lm[PL.LEFT_WRIST.value]
        elbow = lm[PL.LEFT_ELBOW.value]
        shoulder = lm[PL.LEFT_SHOULDER.value]
        ear = lm[PL.LEFT_EAR.value]
    else:
        wrist = lm[PL.RIGHT_WRIST.value]
        elbow = lm[PL.RIGHT_ELBOW.value]
        shoulder = lm[PL.RIGHT_SHOULDER.value]
        ear = lm[PL.RIGHT_EAR.value]

    vis = [wrist.visibility, elbow.visibility, shoulder.visibility]
    if sum(vis) / len(vis) < 0.45:
        return False, "UNKNOWN"

    # 图像坐标：y 越小越高
    if wrist.y < shoulder.y - 0.03:
        # 高于肩，算抬手
        if wrist.y < ear.y:
            return True, "HIGH"
        return True, "RAISED"

    # 手肘已抬高，手腕接近肩，也算半抬
    if elbow.y < shoulder.y and wrist.y < shoulder.y + 0.03:
        return True, "HALF-RAISED"

    return False, "DOWN"


# 手指方向，判别方向和置信度
def finger_direction(hand_landmarks):
    """
    优先根据 食指 MCP(5) -> TIP(8) 的方向判断
    返回:
    - direction_text
    - confidence_text
    """
    if hand_landmarks is None:
        return "NO_HAND", "LOW"

    lm = hand_landmarks.landmark

    # 食指: 5 MCP, 6 PIP, 8 TIP
    mcp = lm[5]
    pip = lm[6]
    tip = lm[8]

    # 先检查食指是否基本伸直
    seg1 = dist2d(mcp, pip)
    seg2 = dist2d(pip, tip)
    direct = dist2d(mcp, tip)

    # 越接近 1 越直
    straight_ratio = direct / (seg1 + seg2 + 1e-6)

    dx = tip.x - mcp.x
    dy = tip.y - mcp.y
    direction = direction_from_vector(dx, dy)

    if straight_ratio > 0.83:
        conf = "HIGH"
    elif straight_ratio > 0.72:
        conf = "MEDIUM"
    else:
        conf = "LOW"

    return direction, conf


#在图上写字
def put_text(img, text, org, color=(0, 255, 0), scale=0.7, thickness=2):
    cv2.putText(
        img, text, org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA
    )


#主程序
def main():
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # 画 landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS
                )

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )

            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )

            body_orientation = "NO_BODY"
            left_raise = "NO_BODY"
            right_raise = "NO_BODY"
            left_finger_dir = "NO_HAND"
            right_finger_dir = "NO_HAND"

            if results.pose_landmarks:
                body_orientation = get_body_orientation(results.pose_landmarks)

                l_up, l_text = is_hand_raised(results.pose_landmarks, "left")
                r_up, r_text = is_hand_raised(results.pose_landmarks, "right")
                left_raise = l_text
                right_raise = r_text

            if results.left_hand_landmarks:
                left_finger_dir, left_conf = finger_direction(results.left_hand_landmarks)
                left_finger_dir = f"{left_finger_dir} ({left_conf})"

            if results.right_hand_landmarks:
                right_finger_dir, right_conf = finger_direction(results.right_hand_landmarks)
                right_finger_dir = f"{right_finger_dir} ({right_conf})"

            # 可视化：在人体附近显示
            y0 = 30
            put_text(frame, f"Body Orientation: {body_orientation}", (20, y0), (0, 255, 0))
            put_text(frame, f"Left Hand Raised: {left_raise}", (20, y0 + 30), (255, 255, 0))
            put_text(frame, f"Right Hand Raised: {right_raise}", (20, y0 + 60), (255, 255, 0))
            put_text(frame, f"Left Index Direction: {left_finger_dir}", (20, y0 + 90), (0, 200, 255))
            put_text(frame, f"Right Index Direction: {right_finger_dir}", (20, y0 + 120), (0, 200, 255))
            put_text(frame, "Press Q to quit", (20, y0 + 160), (0, 0, 255))

            # 在肩膀附近画辅助线，便于观察朝向判断
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                ls = lm[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
                rs = lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
                nose = lm[mp_holistic.PoseLandmark.NOSE.value]

                p_ls = to_px(ls, w, h)
                p_rs = to_px(rs, w, h)
                p_nose = to_px(nose, w, h)
                shoulder_center = ((p_ls[0] + p_rs[0]) // 2, (p_ls[1] + p_rs[1]) // 2)

                cv2.line(frame, p_ls, p_rs, (255, 0, 255), 2)
                cv2.circle(frame, shoulder_center, 5, (255, 255, 255), -1)
                cv2.line(frame, shoulder_center, p_nose, (0, 255, 255), 2)

            cv2.imshow("MediaPipe Orientation + Raise Hand + Finger Direction", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()