import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# 1. 机械臂参数
# =========================
LINK_LENGTHS = np.array([120.0, 100.0, 80.0])   # 三段机械臂长度
NUM_JOINTS = len(LINK_LENGTHS)

# 初始关节角（弧度）
joint_angles = np.radians([10.0, 20.0, -15.0])

# 迭代控制参数
MAX_IK_ITER = 80
POS_TOL = 1e-2
ALPHA = 0.6          # 步长，越大收敛越快，但也可能抖动
LAMBDA = 1e-3        # 阻尼项，避免雅可比矩阵奇异


# =========================
# 2. 正运动学：由关节角算每个关节点位置
# =========================
def forward_kinematics(angles, link_lengths):
    """
    输入:
        angles: [theta1, theta2, theta3]
        link_lengths: [l1, l2, l3]
    输出:
        points: shape=(4,2)
                points[0] = 基座
                points[1] = 第一关节末端
                points[2] = 第二关节末端
                points[3] = 末端执行器
    """
    points = np.zeros((len(angles) + 1, 2))
    total_angle = 0.0
    x, y = 0.0, 0.0

    for i in range(len(angles)):
        total_angle += angles[i]
        x += link_lengths[i] * np.cos(total_angle)
        y += link_lengths[i] * np.sin(total_angle)
        points[i + 1] = [x, y]

    return points


# =========================
# 3. 雅可比矩阵
# =========================
def compute_jacobian(angles, link_lengths):
    """
    对二维平面机械臂末端位置 [x, y] 对各关节角求偏导
    返回 2 x n 的雅可比矩阵
    """
    n = len(angles)
    J = np.zeros((2, n))

    for j in range(n):
        dx = 0.0
        dy = 0.0
        total_angle = 0.0

        for k in range(n):
            total_angle += angles[k]
            if k >= j:
                dx += -link_lengths[k] * np.sin(total_angle)
                dy +=  link_lengths[k] * np.cos(total_angle)

        J[0, j] = dx
        J[1, j] = dy

    return J


# =========================
# 4. 逆运动学：雅可比阻尼最小二乘
# =========================
def inverse_kinematics(target_xy, init_angles, link_lengths,
                       max_iter=MAX_IK_ITER, tol=POS_TOL):
    """
    给定目标末端点，求一组关节角，使机械臂尽量逼近目标
    """
    angles = init_angles.copy()

    for _ in range(max_iter):
        points = forward_kinematics(angles, link_lengths)
        end_effector = points[-1]

        error = target_xy - end_effector
        err_norm = np.linalg.norm(error)

        if err_norm < tol:
            break

        J = compute_jacobian(angles, link_lengths)

        # 阻尼最小二乘:
        # dtheta = J^T (J J^T + λ^2 I)^-1 e
        JJt = J @ J.T
        damping = (LAMBDA ** 2) * np.eye(2)
        dtheta = J.T @ np.linalg.inv(JJt + damping) @ error

        angles += ALPHA * dtheta

        # 角度归一化到 [-pi, pi]
        angles = (angles + np.pi) % (2 * np.pi) - np.pi

    final_points = forward_kinematics(angles, link_lengths)
    final_error = np.linalg.norm(target_xy - final_points[-1])
    return angles, final_points, final_error


# =========================
# 5. 生成一条末端轨迹
#    你可以替换成自己的轨迹点
# =========================
def generate_trajectory():
    """
    生成一个圆形+轻微起伏的轨迹
    你可以替换成你自己的末端轨迹列表
    """
    t = np.linspace(0, 2 * np.pi, 160)
    center = np.array([150.0, 80.0])
    radius = 60.0

    trajectory = []
    for tt in t:
        x = center[0] + radius * np.cos(tt)
        y = center[1] + 0.7 * radius * np.sin(tt)
        trajectory.append(np.array([x, y]))

    return trajectory


# =========================
# 6. 如果你有自己的轨迹，可以这样写
# =========================
def load_custom_trajectory():
    """
    例子：用户自己给定末端点轨迹
    取消注释后可以替换 generate_trajectory()
    """
    trajectory = [
        np.array([180.0,  30.0]),
        np.array([170.0,  50.0]),
        np.array([160.0,  70.0]),
        np.array([150.0,  90.0]),
        np.array([140.0, 110.0]),
        np.array([130.0, 120.0]),
        np.array([120.0, 130.0]),
    ]
    return trajectory


# =========================
# 7. 主流程：逐点跟踪轨迹
# =========================
def simulate():
    global joint_angles

    # 你可以二选一：
    trajectory = generate_trajectory()
    # trajectory = load_custom_trajectory()

    arm_history = []
    end_history = []
    error_history = []
    angle_history = []

    current_angles = joint_angles.copy()

    for i, target in enumerate(trajectory):
        solved_angles, points, err = inverse_kinematics(
            target_xy=target,
            init_angles=current_angles,
            link_lengths=LINK_LENGTHS
        )

        current_angles = solved_angles.copy()

        arm_history.append(points.copy())
        end_history.append(points[-1].copy())
        error_history.append(err)
        angle_history.append(np.degrees(current_angles.copy()))

        print(f"Step {i:03d} | Target=({target[0]:7.2f}, {target[1]:7.2f}) "
              f"| End=({points[-1,0]:7.2f}, {points[-1,1]:7.2f}) "
              f"| Error={err:8.4f} "
              f"| Angles(deg)={np.round(np.degrees(current_angles), 2)}")

    return trajectory, arm_history, end_history, error_history, angle_history


# =========================
# 8. 动画显示
# =========================
def animate_result(trajectory, arm_history, end_history, error_history):
    fig, ax = plt.subplots(figsize=(8, 8))

    total_len = np.sum(LINK_LENGTHS)
    margin = 40

    ax.set_xlim(-total_len - margin, total_len + margin)
    ax.set_ylim(-total_len - margin, total_len + margin)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("2D Robot Arm Trajectory Tracking Simulation")

    # 整体目标轨迹
    traj_xy = np.array(trajectory)
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], '--', linewidth=1.5, label='Target Trajectory')

    # 机械臂
    arm_line, = ax.plot([], [], 'o-', linewidth=3, markersize=8, label='Robot Arm')

    # 当前目标点
    target_point, = ax.plot([], [], 'rx', markersize=10, label='Current Target')

    # 末端实际路径
    end_trace_line, = ax.plot([], [], '-', linewidth=2, label='End Effector Path')

    # 文本信息
    info_text = ax.text(
        0.02, 0.95, '',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    end_path_x = []
    end_path_y = []

    def init():
        arm_line.set_data([], [])
        target_point.set_data([], [])
        end_trace_line.set_data([], [])
        info_text.set_text('')
        return arm_line, target_point, end_trace_line, info_text

    def update(frame):
        pts = arm_history[frame]
        target = trajectory[frame]
        end_pt = end_history[frame]
        err = error_history[frame]

        arm_line.set_data(pts[:, 0], pts[:, 1])
        target_point.set_data([target[0]], [target[1]])

        end_path_x.append(end_pt[0])
        end_path_y.append(end_pt[1])
        end_trace_line.set_data(end_path_x, end_path_y)

        info_text.set_text(
            f'Frame: {frame}\n'
            f'Target: ({target[0]:.1f}, {target[1]:.1f})\n'
            f'End: ({end_pt[0]:.1f}, {end_pt[1]:.1f})\n'
            f'Error: {err:.4f}'
        )

        return arm_line, target_point, end_trace_line, info_text

    ani = FuncAnimation(
        fig, update, frames=len(trajectory),
        init_func=init, interval=60, blit=True, repeat=False
    )

    ax.legend()
    plt.show()


# =========================
# 9. 程序入口
# =========================
if __name__ == "__main__":
    trajectory, arm_history, end_history, error_history, angle_history = simulate()
    animate_result(trajectory, arm_history, end_history, error_history)

    # 输出最终误差统计
    print("\n===== Summary =====")
    print(f"Average Error: {np.mean(error_history):.4f}")
    print(f"Max Error    : {np.max(error_history):.4f}")