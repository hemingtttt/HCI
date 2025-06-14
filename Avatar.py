import cv2
import mediapipe as mp
import socket

"""
实时视频姿态检测与传输脚本：
1. 捕获摄像头图像
2. 使用 MediaPipe Pose 模型检测人体关键点并绘制
3. 提取特定关节的世界坐标
4. 通过 TCP 将坐标字符串发送给 Unity
"""

# 网络配置（在 Unity 中监听的 IP 和端口）
TCP_IP = "127.0.0.1"
TCP_PORT = 12000

# MediaPipe Pose 初始化
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def main():
    # 打开默认摄像头
    cap = cv2.VideoCapture(0)
    # 创建 Pose 模型实例
    with mp_pose.Pose(min_detection_confidence=0.5,
                     min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # 提高处理性能：转换到 RGB 并禁写
            frame.flags.writeable = False
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 执行姿态检测
            results = pose.process(rgb_frame)

            # 恢复写状态并转换回 BGR
            frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # 在画面上绘制姿态关键点与连接线
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            # 水平翻转为自拍视图并显示
            cv2.imshow('MediaPipe Pose', cv2.flip(frame, 1))

            # 如果检测到世界坐标，则提取并发送指定关节位置
            if results.pose_world_landmarks:
                send_joint_positions(results.pose_world_landmarks)

            # 按 ESC 键退出
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


def send_joint_positions(landmarks):
    """
    提取鼻子、左右手腕、左右食指的世界坐标，并通过 TCP 发送。
    坐标格式："x,y,z;x,y,z;..."
    """
    # 关节点索引：0=Nose,15=LeftWrist,16=RightWrist,19=LeftIndex,20=RightIndex
    joints = [0, 15, 16, 19, 20]
    coords = []
    for idx in joints:
        lm = landmarks.landmark[idx]
        coords.append(f"{lm.x},{lm.y},{lm.z}")
    message = ';'.join(coords)

    # 发送 TCP 数据
    with socket.socket() as sock:
        sock.connect((TCP_IP, TCP_PORT))
        sock.sendall(message.encode('utf-8'))


if __name__ == '__main__':
    main()


'''
这段代码实现了一个实时姿态检测和数据传输系统：

通过摄像头捕获视频流
使用 MediaPipe 进行姿态检测，识别 33 个关键点
在图像上绘制姿态骨架
提取五个关键关节（鼻子、左右手腕、左右食指）的 3D 坐标
将坐标数据格式化为字符串，通过 TCP 发送到 Unity 应用
提供可视化界面，显示处理后的视频
'''