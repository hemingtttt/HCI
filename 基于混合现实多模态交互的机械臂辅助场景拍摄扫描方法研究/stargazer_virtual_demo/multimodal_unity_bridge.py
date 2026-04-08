from __future__ import annotations

import json
import math
import queue
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------- import your multimodal project ----------
# Place this file next to the extracted stargazer_virtual_demo folder,
# or edit PROJECT_ROOT below.
PROJECT_ROOT = Path(__file__).resolve().parent / "stargazer_virtual_demo"
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DemoConfig
from src.controller.multimodal_fusion import MultimodalFusion
from src.sensors.gesture_pose_voice_en_ch_overlay_hold import GesturePoseVoiceSensor


# =========================================================
# 1. TCP sender: keep the JSON protocol identical to Unity side
# =========================================================
class UnitySender:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000) -> None:
        self.host = host
        self.port = port
        self.server: Optional[socket.socket] = None
        self.conn: Optional[socket.socket] = None
        self.addr = None

    def wait_for_unity(self) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        print(f"[Bridge] Waiting for Unity on {self.host}:{self.port} ...")
        self.conn, self.addr = self.server.accept()
        print(f"[Bridge] Unity connected from {self.addr}")

    def send_state(self, payload: dict) -> None:
        if self.conn is None:
            raise RuntimeError("Unity is not connected.")
        msg = json.dumps(payload, ensure_ascii=False) + "\n"
        self.conn.sendall(msg.encode("utf-8"))

    def close(self) -> None:
        try:
            if self.conn is not None:
                self.conn.close()
        finally:
            self.conn = None
            if self.server is not None:
                self.server.close()
                self.server = None


# =========================================================
# 2. Robot arm kinematics + DLS IK
#    (same logic as your earlier mechanical_arm_3d.py)
# =========================================================
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


def rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def roty(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    q1, q2, q3, q4, q5 = q
    points = []
    p0 = np.array([0.0, 0.0, 0.0])
    points.append(p0)
    p1 = np.array([0.0, 0.0, L1])
    points.append(p1)
    r_base = rotz(q1)
    r2 = r_base @ roty(q2)
    p2 = p1 + r2 @ np.array([L2, 0.0, 0.0])
    points.append(p2)
    r3 = r2 @ roty(q3)
    p3 = p2 + r3 @ np.array([L3, 0.0, 0.0])
    points.append(p3)
    r4 = r3 @ roty(q4)
    p4 = p3 + r4 @ np.array([L4, 0.0, 0.0])
    points.append(p4)
    return np.array(points)


def end_effector_position(q: np.ndarray) -> np.ndarray:
    return forward_kinematics(q)[-1]


def numerical_jacobian(q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    j = np.zeros((3, len(q)))
    p0 = end_effector_position(q)
    for i in range(len(q)):
        q_eps = q.copy()
        q_eps[i] += eps
        p1 = end_effector_position(q_eps)
        j[:, i] = (p1 - p0) / eps
    return j


def clamp_joints(q: np.ndarray) -> np.ndarray:
    q2 = q.copy()
    for i in range(len(q2)):
        q2[i] = np.clip(q2[i], joint_limits[i, 0], joint_limits[i, 1])
    return q2


def inverse_kinematics(target: np.ndarray, q_start: np.ndarray) -> tuple[np.ndarray, float]:
    q = q_start.copy()
    for _ in range(MAX_ITER):
        p = end_effector_position(q)
        error = target - p
        if np.linalg.norm(error) < POS_TOL:
            break
        j = numerical_jacobian(q)
        jjt = j @ j.T
        damping = (LAMBDA ** 2) * np.eye(3)
        dq = j.T @ np.linalg.inv(jjt + damping) @ error
        q = clamp_joints(q + ALPHA * dq)
    final_error = np.linalg.norm(target - end_effector_position(q))
    return q, float(final_error)


# =========================================================
# 3. Semantic camera command -> reachable robot target
#    Replace the old pre-generated trajectory with a live target.
# =========================================================
@dataclass
class SemanticCameraState:
    radius_mm: float = 90.0
    azimuth_deg: float = 25.0
    elevation_deg: float = 18.0
    subject_center_mm: tuple[float, float, float] = (120.0, 0.0, 110.0)


class SemanticTargetBridge:
    def __init__(
        self,
        close_radius_mm: float = 60.0,
        medium_radius_mm: float = 90.0,
        level_elevation_deg: float = 18.0,
        top_elevation_deg: float = 58.0,
        orbit_step_deg: float = 3.0,
        pan_step_deg: float = 4.0,
        azimuth_min_deg: float = -75.0,
        azimuth_max_deg: float = 75.0,
    ) -> None:
        self.close_radius_mm = close_radius_mm
        self.medium_radius_mm = medium_radius_mm
        self.level_elevation_deg = level_elevation_deg
        self.top_elevation_deg = top_elevation_deg
        self.orbit_step_deg = orbit_step_deg
        self.pan_step_deg = pan_step_deg
        self.azimuth_min_deg = azimuth_min_deg
        self.azimuth_max_deg = azimuth_max_deg
        self.state = SemanticCameraState(
            radius_mm=medium_radius_mm,
            azimuth_deg=25.0,
            elevation_deg=level_elevation_deg,
        )

    def apply_command(self, cmd: str) -> None:
        if cmd == "top":
            self.state.elevation_deg = self.top_elevation_deg
        elif cmd == "level":
            self.state.elevation_deg = self.level_elevation_deg
        elif cmd == "close":
            self.state.radius_mm = self.close_radius_mm
        elif cmd == "medium":
            self.state.radius_mm = self.medium_radius_mm
        elif cmd == "orbit_left":
            self.state.azimuth_deg -= self.orbit_step_deg
        elif cmd == "orbit_right":
            self.state.azimuth_deg += self.orbit_step_deg
        elif cmd == "pan_left":
            self.state.azimuth_deg -= self.pan_step_deg
        elif cmd == "pan_right":
            self.state.azimuth_deg += self.pan_step_deg
        elif cmd == "reset":
            self.state.radius_mm = self.medium_radius_mm
            self.state.azimuth_deg = 25.0
            self.state.elevation_deg = self.level_elevation_deg

        self.state.azimuth_deg = float(np.clip(self.state.azimuth_deg, self.azimuth_min_deg, self.azimuth_max_deg))

    def current_target_mm(self) -> np.ndarray:
        cx, cy, cz = self.state.subject_center_mm
        az = math.radians(self.state.azimuth_deg)
        el = math.radians(self.state.elevation_deg)
        r = self.state.radius_mm
        x = cx + r * math.cos(el) * math.cos(az)
        y = cy + r * math.cos(el) * math.sin(az)
        z = cz + r * math.sin(el)
        return np.array([x, y, z], dtype=float)


# =========================================================
# 4. Coordinate conversion for Unity
# =========================================================
SCALE_TO_UNITY = 0.01  # mm -> Unity units


def py_to_unity_vec3(p_mm: np.ndarray) -> list[float]:
    # Python arm: x, y, z(up)
    # Unity:      x, y(up), z
    return [float(p_mm[0] * SCALE_TO_UNITY), float(p_mm[2] * SCALE_TO_UNITY), float(p_mm[1] * SCALE_TO_UNITY)]


# =========================================================
# 5. Helper: terminal text input for mock mode
# =========================================================
def spawn_mock_input_thread(command_queue: "queue.Queue[str]") -> threading.Thread:
    def _worker() -> None:
        while True:
            try:
                text = input().strip().lower()
            except EOFError:
                return
            if text:
                command_queue.put(text)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


# =========================================================
# 6. Main bridge app
# =========================================================
def main() -> None:
    cfg = DemoConfig()
    fusion = MultimodalFusion(hold_seconds=cfg.fusion_hold_seconds)
    sensor = GesturePoseVoiceSensor(camera_index=cfg.camera_index)

    semantic_bridge = SemanticTargetBridge(
        close_radius_mm=60.0,
        medium_radius_mm=90.0,
        level_elevation_deg=cfg.level_elevation_deg,
        top_elevation_deg=58.0,
        orbit_step_deg=cfg.orbit_step_deg,
        pan_step_deg=cfg.pan_step_deg,
    )

    sender = UnitySender(host="127.0.0.1", port=9000)
    sender.wait_for_unity()

    cmd_queue: "queue.Queue[str]" = queue.Queue()
    spawn_mock_input_thread(cmd_queue)
    sensor.start()

    q_current = q_init.copy()
    frame_idx = 0
    last_send_time = 0.0
    active_command = "reset"
    app_start_time = time.time()

    print("[Bridge] Started. Press q to quit, m to toggle mock mode.")
    print("[Bridge] Mock commands: top, level, close, medium, orbit_left, orbit_right, pan_left, pan_right, reset")

    try:
        while True:
            while not cmd_queue.empty():
                sensor.set_mock_command(cmd_queue.get_nowait())

            result = sensor.read()
            active_command = fusion.fuse(result.voice, result.gesture, result.pose)
            semantic_bridge.apply_command(active_command)
            target_mm = semantic_bridge.current_target_mm()

            q_sol, err = inverse_kinematics(target_mm, q_current)
            q_current = q_sol.copy()
            pts = forward_kinematics(q_current)
            ee_mm = pts[-1]

            now = time.time()
            if now - last_send_time >= cfg.render_interval_s:
                payload = {
                    "type": "state",
                    "frame": frame_idx,
                    "time": now - app_start_time,
                    "q_deg": [float(v) for v in np.degrees(q_current)],
                    "target": py_to_unity_vec3(target_mm),
                    "ee": py_to_unity_vec3(ee_mm),
                    "err": float(err),
                }
                sender.send_state(payload)
                last_send_time = now
                frame_idx += 1

            # keep the original camera window for debugging the perception side
            frame = result.frame
            if frame is not None:
                overlay = frame.copy()
                cv2.putText(overlay, f"cmd={active_command}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(overlay, f"target_mm={np.round(target_mm, 1)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(overlay, f"q_deg={np.round(np.degrees(q_current), 1)}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(overlay, f"err={err:.2f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.imshow("Multimodal -> Unity Bridge", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                is_mock = sensor.toggle_mock_mode()
                print(f"[Bridge] mock_mode={is_mock}")

    except (BrokenPipeError, ConnectionResetError):
        print("[Bridge] Unity disconnected.")
    finally:
        sensor.stop()
        sender.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
