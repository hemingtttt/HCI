import json
import socket
import time
import numpy as np

# =========================
# 1. 机械臂参数（与你原代码一致）
# =========================
L1 = 80.0
L2 = 120.0
L3 = 100.0
L4 = 60.0

q_init = np.radians([0.0, 20.0, -30.0, 10.0, 0.0])

joint_limits_deg = [
    (-180, 180),
    (-90, 90),
    (-120, 120),
    (-120, 120),
    (-180, 180),
]
joint_limits = np.radians(np.array(joint_limits_deg, dtype=float))

MAX_ITER = 100
POS_TOL = 1e-2
ALPHA = 0.5
LAMBDA = 1e-2

# 发送频率
FPS = 30.0
DT = 1.0 / FPS

# 单位缩放：mm -> Unity units
SCALE = 0.01

HOST = "127.0.0.1"
PORT = 9000


# =========================
# 2. 基础旋转矩阵
# =========================
def rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def roty(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])


# =========================
# 3. 正运动学
# =========================
def forward_kinematics(q):
    q1, q2, q3, q4, q5 = q

    points = []

    p0 = np.array([0.0, 0.0, 0.0])
    points.append(p0)

    p1 = np.array([0.0, 0.0, L1])
    points.append(p1)

    R_base = rotz(q1)

    R2 = R_base @ roty(q2)
    p2 = p1 + R2 @ np.array([L2, 0.0, 0.0])
    points.append(p2)

    R3 = R2 @ roty(q3)
    p3 = p2 + R3 @ np.array([L3, 0.0, 0.0])
    points.append(p3)

    R4 = R3 @ roty(q4)
    p4 = p3 + R4 @ np.array([L4, 0.0, 0.0])
    points.append(p4)

    return np.array(points)

def end_effector_position(q):
    return forward_kinematics(q)[-1]


# =========================
# 4. 数值雅可比
# =========================
def numerical_jacobian(q, eps=1e-5):
    J = np.zeros((3, len(q)))
    p0 = end_effector_position(q)

    for i in range(len(q)):
        q_eps = q.copy()
        q_eps[i] += eps
        p1 = end_effector_position(q_eps)
        J[:, i] = (p1 - p0) / eps

    return J


# =========================
# 5. 限位
# =========================
def clamp_joints(q):
    q_clamped = q.copy()
    for i in range(len(q)):
        q_clamped[i] = np.clip(q_clamped[i], joint_limits[i, 0], joint_limits[i, 1])
    return q_clamped


# =========================
# 6. 阻尼最小二乘 IK
# =========================
def inverse_kinematics(target, q_start):
    q = q_start.copy()

    for _ in range(MAX_ITER):
        p = end_effector_position(q)
        error = target - p
        err_norm = np.linalg.norm(error)

        if err_norm < POS_TOL:
            break

        J = numerical_jacobian(q)
        JJt = J @ J.T
        damping = (LAMBDA ** 2) * np.eye(3)
        dq = J.T @ np.linalg.inv(JJt + damping) @ error

        q = q + ALPHA * dq
        q = clamp_joints(q)

    final_p = end_effector_position(q)
    final_error = np.linalg.norm(target - final_p)
    return q, final_error


# =========================
# 7. 目标轨迹
# =========================
def generate_3d_trajectory():
    trajectory = []
    t_vals = np.linspace(0, 2 * np.pi, 180)

    center = np.array([140.0, 0.0, 120.0])
    rx = 50.0
    ry = 70.0
    rz = 30.0

    for t in t_vals:
        x = center[0] + rx * np.cos(t)
        y = center[1] + ry * np.sin(t)
        z = center[2] + rz * np.sin(2 * t)
        trajectory.append(np.array([x, y, z], dtype=float))

    return trajectory


# =========================
# 8. 坐标转换：Python -> Unity
# Python: x, y, z(up)
# Unity : x, y(up), z
# =========================
def py_to_unity_vec3(p):
    return [float(p[0] * SCALE), float(p[2] * SCALE), float(p[1] * SCALE)]


# =========================
# 9. 发送 JSON（一行一条）
# =========================
def send_json_line(conn, obj):
    msg = json.dumps(obj, ensure_ascii=False) + "\n"
    conn.sendall(msg.encode("utf-8"))


# =========================
# 10. 主服务
# =========================
def run_server():
    trajectory = generate_3d_trajectory()
    q_current = q_init.copy()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"[Python] Waiting for Unity on {HOST}:{PORT} ...")
    conn, addr = server.accept()
    print(f"[Python] Unity connected from {addr}")

    start_time = time.time()

    try:
        for i, target in enumerate(trajectory):
            frame_begin = time.time()

            q_sol, err = inverse_kinematics(target, q_current)
            q_current = q_sol.copy()

            pts = forward_kinematics(q_current)
            ee = pts[-1]

            msg = {
                "type": "state",
                "frame": i,
                "time": time.time() - start_time,
                "q_deg": [float(v) for v in np.degrees(q_current)],
                "target": py_to_unity_vec3(target),
                "ee": py_to_unity_vec3(ee),
                "err": float(err)
            }

            send_json_line(conn, msg)

            print(
                f"[Python] frame={i:03d} "
                f"target={np.round(target, 2)} "
                f"ee={np.round(ee, 2)} "
                f"q_deg={np.round(np.degrees(q_current), 2)} "
                f"err={err:.4f}"
            )

            elapsed = time.time() - frame_begin
            sleep_time = max(0.0, DT - elapsed)
            time.sleep(sleep_time)

    except (BrokenPipeError, ConnectionResetError):
        print("[Python] Unity disconnected.")
    finally:
        conn.close()
        server.close()


if __name__ == "__main__":
    run_server()