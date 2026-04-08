import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# =========================================================
# 1. 参数区
# =========================================================

# 连杆长度（单位你可以理解成 mm）
L1 = 80.0    # 基座到肩部的高度
L2 = 120.0   # 大臂
L3 = 100.0   # 小臂
L4 = 60.0    # 腕部/末端段

# 初始关节角（弧度）
# q1: base yaw
# q2: shoulder pitch
# q3: elbow pitch
# q4: wrist pitch
# q5: reserved/tool compensation
q_init = np.radians([0.0, 20.0, -30.0, 10.0, 0.0])

# 关节限位（可改）
joint_limits_deg = [
    (-180, 180),   # q1
    (-90, 90),     # q2
    (-120, 120),   # q3
    (-120, 120),   # q4
    (-180, 180),   # q5
]
joint_limits = np.radians(np.array(joint_limits_deg, dtype=float))

# 逆运动学参数
MAX_ITER = 100
POS_TOL = 1e-2
ALPHA = 0.5
LAMBDA = 1e-2

# 仿真帧间隔
INTERVAL_MS = 60


# =========================================================
# 2. 基础旋转矩阵
# =========================================================

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


# =========================================================
# 3. 正运动学
# =========================================================
# 返回所有关节点的三维位置
#
# 机械臂结构可理解为：
# base --(z向上L1)--> shoulder
# shoulder 经 q2 后沿局部x伸出 L2
# elbow    经 q3 后沿局部x伸出 L3
# wrist    经 q4 后沿局部x伸出 L4
# q1 决定整个机械臂在水平面朝向
#
# q5 这里保留，不影响末端位置，只作为后续扩展接口
# =========================================================

def forward_kinematics(q):
    q1, q2, q3, q4, q5 = q

    points = []

    # 基座原点
    p0 = np.array([0.0, 0.0, 0.0])
    points.append(p0)

    # 到肩部：沿全局z轴抬高 L1
    p1 = np.array([0.0, 0.0, L1])
    points.append(p1)

    # 总方向
    R_base = rotz(q1)

    # 第一段（大臂）
    R2 = R_base @ roty(q2)
    p2 = p1 + R2 @ np.array([L2, 0.0, 0.0])
    points.append(p2)

    # 第二段（小臂）
    R3 = R2 @ roty(q3)
    p3 = p2 + R3 @ np.array([L3, 0.0, 0.0])
    points.append(p3)

    # 第三段（腕部/工具段）
    R4 = R3 @ roty(q4)
    p4 = p3 + R4 @ np.array([L4, 0.0, 0.0])
    points.append(p4)

    return np.array(points)  # shape = (5, 3)


def end_effector_position(q):
    return forward_kinematics(q)[-1]


# =========================================================
# 4. 数值雅可比矩阵
# =========================================================
# 用有限差分，简单直观，便于初学者理解
# 只关心末端位置 x,y,z 对各关节角的偏导
# 输出 3x5
# =========================================================

def numerical_jacobian(q, eps=1e-5):
    J = np.zeros((3, len(q)))
    p0 = end_effector_position(q)

    for i in range(len(q)):
        q_eps = q.copy()
        q_eps[i] += eps
        p1 = end_effector_position(q_eps)
        J[:, i] = (p1 - p0) / eps

    return J


# =========================================================
# 5. 关节限位裁剪
# =========================================================

def clamp_joints(q):
    q_clamped = q.copy()
    for i in range(len(q)):
        q_clamped[i] = np.clip(q_clamped[i], joint_limits[i, 0], joint_limits[i, 1])
    return q_clamped


# =========================================================
# 6. 逆运动学：阻尼最小二乘
# =========================================================
# 目标：让末端位置逼近 target = [x, y, z]
# =========================================================

def inverse_kinematics(target, q_start):
    q = q_start.copy()

    for _ in range(MAX_ITER):
        p = end_effector_position(q)
        error = target - p
        err_norm = np.linalg.norm(error)

        if err_norm < POS_TOL:
            break

        J = numerical_jacobian(q)

        # 阻尼最小二乘
        # dq = J^T (J J^T + λ²I)^(-1) e
        JJt = J @ J.T
        damping = (LAMBDA ** 2) * np.eye(3)
        dq = J.T @ np.linalg.inv(JJt + damping) @ error

        q = q + ALPHA * dq
        q = clamp_joints(q)

    final_p = end_effector_position(q)
    final_error = np.linalg.norm(target - final_p)
    return q, final_error


# =========================================================
# 7. 生成一条 3D 目标轨迹
# =========================================================
# 你后面可以替换成你自己的末端轨迹
# =========================================================

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


# =========================================================
# 8. 如果你想手工输入自己的轨迹
# =========================================================

def custom_trajectory():
    return [
        np.array([180.0,   0.0, 100.0]),
        np.array([170.0,  20.0, 110.0]),
        np.array([160.0,  40.0, 120.0]),
        np.array([150.0,  60.0, 130.0]),
        np.array([140.0,  80.0, 120.0]),
        np.array([130.0,  60.0, 110.0]),
        np.array([120.0,  40.0, 100.0]),
    ]


# =========================================================
# 9. 主仿真流程
# =========================================================

def run_simulation():
    # 你可以二选一
    trajectory = generate_3d_trajectory()
    # trajectory = custom_trajectory()

    q_current = q_init.copy()

    q_history = []
    arm_points_history = []
    ee_history = []
    err_history = []

    for i, target in enumerate(trajectory):
        q_sol, err = inverse_kinematics(target, q_current)
        q_current = q_sol.copy()

        pts = forward_kinematics(q_current)
        ee = pts[-1]

        q_history.append(q_current.copy())
        arm_points_history.append(pts.copy())
        ee_history.append(ee.copy())
        err_history.append(err)

        print(
            f"Step {i:03d} | "
            f"Target=({target[0]:7.2f}, {target[1]:7.2f}, {target[2]:7.2f}) | "
            f"EE=({ee[0]:7.2f}, {ee[1]:7.2f}, {ee[2]:7.2f}) | "
            f"Err={err:8.4f} | "
            f"q(deg)={np.round(np.degrees(q_current), 2)}"
        )

    return trajectory, q_history, arm_points_history, ee_history, err_history


# =========================================================
# 10. 3D 动画显示
# =========================================================

def animate_simulation(trajectory, arm_points_history, ee_history, err_history):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 工作空间范围
    arm_max = L1 + L2 + L3 + L4 + 40
    ax.set_xlim(-arm_max, arm_max)
    ax.set_ylim(-arm_max, arm_max)
    ax.set_zlim(0, arm_max)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Robot Arm Trajectory Tracking")

    # 目标轨迹
    traj_np = np.array(trajectory)
    ax.plot(traj_np[:, 0], traj_np[:, 1], traj_np[:, 2], '--', linewidth=1.5, label='Target Trajectory')

    # 当前机械臂
    arm_line, = ax.plot([], [], [], 'o-', linewidth=3, markersize=6, label='Robot Arm')

    # 当前目标点
    target_scatter = ax.scatter([], [], [], marker='x', s=80, label='Current Target')

    # 末端实际轨迹
    ee_line, = ax.plot([], [], [], linewidth=2, label='End Effector Path')

    # 文本信息
    text_info = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    ee_path_x, ee_path_y, ee_path_z = [], [], []

    def init():
        arm_line.set_data([], [])
        arm_line.set_3d_properties([])
        ee_line.set_data([], [])
        ee_line.set_3d_properties([])
        text_info.set_text("")
        return arm_line, ee_line, text_info

    def update(frame):
        nonlocal target_scatter

        pts = arm_points_history[frame]
        target = trajectory[frame]
        ee = ee_history[frame]
        err = err_history[frame]

        # 更新机械臂折线
        arm_line.set_data(pts[:, 0], pts[:, 1])
        arm_line.set_3d_properties(pts[:, 2])

        # 更新当前目标点
        target_scatter.remove()
        target_scatter = ax.scatter(
            [target[0]], [target[1]], [target[2]],
            marker='x', s=80
        )

        # 更新末端轨迹
        ee_path_x.append(ee[0])
        ee_path_y.append(ee[1])
        ee_path_z.append(ee[2])
        ee_line.set_data(ee_path_x, ee_path_y)
        ee_line.set_3d_properties(ee_path_z)

        text_info.set_text(
            f"Frame: {frame}\n"
            f"Target: ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})\n"
            f"EE: ({ee[0]:.1f}, {ee[1]:.1f}, {ee[2]:.1f})\n"
            f"Error: {err:.4f}"
        )

        return arm_line, ee_line, text_info

    ani = FuncAnimation(
        fig,
        update,
        frames=len(trajectory),
        init_func=init,
        interval=INTERVAL_MS,
        blit=False,
        repeat=False
    )

    ax.legend()
    plt.show()


# =========================================================
# 11. 程序入口
# =========================================================

if __name__ == "__main__":
    trajectory, q_history, arm_points_history, ee_history, err_history = run_simulation()
    animate_simulation(trajectory, arm_points_history, ee_history, err_history)

    print("\n===== Summary =====")
    print(f"Average Error: {np.mean(err_history):.4f}")
    print(f"Max Error    : {np.max(err_history):.4f}")
