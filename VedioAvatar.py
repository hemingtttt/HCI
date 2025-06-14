import cv2
import mediapipe as mp
import numpy as np
import socket

# === 配置 ===
TCP_IP = "127.0.0.1"
TCP_PORT = 12000
BG_COLOR = (0, 0, 0, 0)  # 透明黑背景

# === MediaPipe 初始化 ===
mp_selfie = mp.solutions.selfie_segmentation

# === 摄像头与模型 ===
cap = cv2.VideoCapture(0)
with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:
    bg_image = None

    while cap.isOpened():
        # 读取并预处理图像
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        # 人像分割
        results = segmenter.process(frame)

        # 恢复图像写权限，并转回 BGRA
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)

        # 初始化透明背景
        if bg_image is None:
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR

        # 根据分割结果合成前景与背景
        mask = (results.segmentation_mask > 0.1)
        mask = np.stack((mask,)*4, axis=-1)
        output = np.where(mask, frame, bg_image)

        # 显示用于调试（可删除）
        cv2.imshow('Segmentation', output)

        # 将图像编码为 PNG 并通过 TCP 发送
        _, buf = cv2.imencode('.png', output, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        data = buf.tobytes()
        with socket.socket() as sock:
            sock.connect((TCP_IP, TCP_PORT))
            sock.sendall(data)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()


'''
这段代码实现了一个实时人像分割与透明背景传输系统：

通过摄像头捕获视频流
使用 MediaPipe 的自拍分割模型进行前景 / 背景分离
将背景替换为透明色（RGBA 格式）并合成新图像
将处理后的图像编码为 PNG 格式
通过 TCP 套接字将图像数据发送到Unity
提供可视化界面，实时显示分割效果
'''