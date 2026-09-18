#!/usr/bin/env python3
"""
UF850 + OpenPI pi0.5 — minimal single-threaded inference loop.

Same pattern as official OpenPI examples (examples/droid/main.py):
  1. Read observation (joints, gripper, cameras, optional tactile)
  2. Send to policy server -> receive action chunk
  3. Execute chunk step-by-step at CTRL_HZ
  4. Repeat
"""

import os
import time
import logging
import threading
from collections import deque
from typing import Optional

import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
from openpi_client import websocket_client_policy as wcp
from openpi_client import image_tools

logging.basicConfig(level=logging.INFO, force=True)
log = logging.getLogger("uf850_loop")

# ─── Configuration ───────────────────────────────────────────────────────────

SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))
ROBOT_IP = os.environ.get("ROBOT_IP", "192.168.1.117")

MAIN_CAM_SERIAL = os.environ.get("MAIN_RS_SERIAL", "216322072605")
WRIST_CAM_SERIAL = os.environ.get("WRIST_RS_SERIAL", "148522073633")

PROMPT = os.environ.get(
    "PROMPT",
    "grasp the test tube and insert it into the target position on the rack",
)

CTRL_HZ = int(os.environ.get("CTRL_HZ", "20"))
OPEN_LOOP_HORIZON = int(os.environ.get("OPEN_LOOP_HORIZON", "8"))

RS_W, RS_H, RS_FPS = 640, 480, 30
IMG_SIZE = 224

SERVO_SPEED = int(os.environ.get("SERVO_SPEED", "120"))
SERVO_MVACC = int(os.environ.get("SERVO_MVACC", "2000"))

GRIP_MAX = 850
GRIP_MIN = float(os.environ.get("GRIP_MIN", "45"))
OPEN_TH = float(os.environ.get("GRIP_OPEN_TH", "0.965")) * GRIP_MAX
CLOSE_TH = float(os.environ.get("GRIP_CLOSE_TH", "0.2")) * GRIP_MAX
GRIP_COOLDOWN = float(os.environ.get("GRIP_STATE_COOLDOWN_S", "0.3"))

# FSR / tactile (optional — set FSR_PORT env var to enable)
FSR_PORT = os.environ.get("FSR_PORT", "/dev/ttyACM0")
FSR_BAUD = int(os.environ.get("FSR_BAUD", "500000"))
FSR_MODE = int(os.environ.get("FSR_MODE", "1"))
FSR_H = int(os.environ.get("FSR_H", "16"))
FSR_W = int(os.environ.get("FSR_W", "16"))
FSR_T = int(os.environ.get("FSR_T", "5"))
FSR_BASELINE_COUNT = int(os.environ.get("FSR_BASELINE_COUNT", "120"))
FSR_KEEP_LAST_N = int(os.environ.get("FSR_KEEP_LAST_N", "200"))
FSR_SCALE_DIV = float(os.environ.get("FSR_SCALE_DIV", "50.0"))
# When FSR is enabled, wait for tactile readiness before inference.
# Readiness means the baseline is computed and a full window is available.
# Timeout is in seconds; 0 disables waiting.
FSR_WAIT_TIMEOUT = float(os.environ.get("FSR_WAIT_TIMEOUT", "60.0"))


# ─── RealSense Camera ───────────────────────────────────────────────────────

class RSCamera:
    def __init__(self, serial_no: str, name: str):
        self.serial = serial_no
        self.name = name
        self.pipeline = None

    def start(self):
        ctx = rs.context()
        deadline = time.time() + 8.0
        found = False
        while time.time() < deadline:
            for d in ctx.query_devices():
                if d.get_info(rs.camera_info.serial_number) == self.serial:
                    found = True
                    break
            if found:
                break
            time.sleep(0.2)
        if not found:
            raise RuntimeError(f"[{self.name}] Camera {self.serial} not found")

        p = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
        p.start(cfg)
        for _ in range(10):
            p.poll_for_frames()
        self.pipeline = p
        log.info(f"Camera [{self.name}] started (serial={self.serial})")

    def read_rgb(self) -> np.ndarray:
        if self.pipeline is None:
            self.start()
        deadline = time.time() + 1.0
        while time.time() < deadline:
            frames = self.pipeline.poll_for_frames()
            if frames:
                cf = frames.get_color_frame()
                if cf:
                    bgr = np.asanyarray(cf.get_data())
                    rgb = bgr[..., ::-1].copy()
                    rgb = image_tools.resize_with_pad(rgb, IMG_SIZE, IMG_SIZE)
                    return image_tools.convert_to_uint8(rgb)
            time.sleep(0.002)
        raise RuntimeError(f"[{self.name}] frame timeout")

    def stop(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None


# ─── FSR Tactile Reader (sensor I/O driver with internal serial thread) ──────

class FSRReader:
    def __init__(self, port, baudrate, mode, size,
                 baseline_count, keep_last_n, scale_div):
        import serial as _serial

        self.mode = mode
        self.size = size
        self.scale_div = scale_div
        self.reorder = [8, 9, 10, 11, 12, 13, 14, 15, 7, 6, 5, 4, 3, 2, 1, 0]

        self._baseline_count = baseline_count
        self._bl1_buf, self._bl2_buf = [], []
        self._bl1, self._bl2 = None, None

        self._lock = threading.Lock()
        self._frames = deque(maxlen=keep_last_n)
        self._stop_evt = threading.Event()

        self.ser = _serial.Serial(port, baudrate, timeout=3)
        time.sleep(1.0)
        self.ser.reset_input_buffer()

        threading.Thread(target=self._loop, daemon=True).start()
        log.info(f"FSR started on {port} (mode={mode})")

    def _apply_baseline(self, raw, buf, bl, tag):
        buf.append(raw.copy())
        if bl is None and len(buf) >= self._baseline_count:
            bl = np.mean(buf, axis=0)
            log.info(f"FSR {tag} baseline computed")
        if bl is not None:
            return np.clip(raw - bl, 0, None) / self.scale_div, bl
        return raw, bl

    def _loop(self):
        cmd = ((self.mode << 6) | self.size).to_bytes(1, "big")
        tlen = 4
        flen = (2 if self.mode == 0 else 4) * self.size ** 2
        dlen = flen + tlen

        while not self._stop_evt.is_set():
            try:
                self.ser.write(cmd)
                d = self.ser.read(dlen)
                if len(d) != dlen:
                    continue

                frame = d[:-tlen]
                if self.mode == 0:
                    raw = np.frombuffer(frame, np.uint16).reshape(self.size, self.size)
                    r1, r2 = raw, None
                else:
                    raw = np.frombuffer(frame, np.uint16).reshape(self.size, self.size * 2)
                    r1, r2 = raw[:, :self.size], raw[:, self.size:]

                f1 = r1[self.reorder].astype(np.float32)
                f1, self._bl1 = self._apply_baseline(f1, self._bl1_buf, self._bl1, "L")

                f2 = None
                if r2 is not None:
                    f2 = r2[self.reorder].astype(np.float32)
                    f2, self._bl2 = self._apply_baseline(f2, self._bl2_buf, self._bl2, "R")

                valid = self._bl1 is not None and (self.mode == 0 or self._bl2 is not None)

                with self._lock:
                    self._frames.append({"fsr1": f1, "fsr2": f2, "valid": valid})
            except Exception as e:
                log.warning(f"FSR: {e}")
                time.sleep(0.05)

    def get_window(self, t: int):
        with self._lock:
            valid = [f for f in self._frames if f["valid"]]
        if len(valid) < t:
            return None, None
        frames = valid[-t:]
        left = np.stack([f["fsr1"] for f in frames]).astype(np.float32)
        if all(f["fsr2"] is not None for f in frames):
            right = np.stack([f["fsr2"] for f in frames]).astype(np.float32)
        else:
            right = np.zeros_like(left)
        return left, right

    def stop(self):
        self._stop_evt.set()
        try:
            self.ser.close()
        except Exception:
            pass


# ─── Gripper Helpers ─────────────────────────────────────────────────────────

class GripperState:
    """Binarized gripper state with hysteresis (for model input)."""

    def __init__(self):
        self.g_bin = 0.0
        self._last_flip = 0.0

    def update(self, pos: float, now: float) -> float:
        if now - self._last_flip < GRIP_COOLDOWN:
            return self.g_bin
        if self.g_bin < 0.5 and pos >= OPEN_TH:
            self.g_bin = 1.0
            self._last_flip = now
        elif self.g_bin >= 0.5 and pos <= CLOSE_TH:
            self.g_bin = 0.0
            self._last_flip = now
        return self.g_bin


def read_gripper(arm: XArmAPI) -> float:
    try:
        ret = arm.get_gripper_position()
        if isinstance(ret, (list, tuple)) and len(ret) >= 2 and ret[0] == 0:
            return float(ret[1])
    except Exception:
        pass
    return 0.0


def send_gripper(arm: XArmAPI, g_01: float):
    pos = float(np.clip(g_01, 0, 1)) * GRIP_MAX
    pos = float(np.clip(pos, GRIP_MIN, GRIP_MAX))
    try:
        arm.set_gripper_position(pos, wait=False)
    except Exception:
        pass


# ─── Main Loop ───────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  UF850 + pi0.5 — Simple Inference Loop")
    print("=" * 55)
    print(f"  Server:  {SERVER_HOST}:{SERVER_PORT}")
    print(f"  Robot:   {ROBOT_IP}")
    print(f"  Prompt:  {PROMPT}")
    print(f"  Ctrl Hz: {CTRL_HZ}   Open-loop horizon: {OPEN_LOOP_HORIZON}")
    tac_str = f"ENABLED ({FSR_PORT})" if FSR_PORT.strip() else "DISABLED"
    print(f"  Tactile: {tac_str}")
    print("=" * 55)

    # ── Robot ──
    arm = XArmAPI(ROBOT_IP, is_radian=True)
    arm.connect()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)

    # ── Policy server (sync client, same as official examples) ──
    client = wcp.WebsocketClientPolicy(host=SERVER_HOST, port=SERVER_PORT)
    log.info(f"Connected. Server metadata: {client.get_server_metadata()}")

    # ── Cameras ──
    cam_main = RSCamera(MAIN_CAM_SERIAL, "MAIN")
    cam_wrist = RSCamera(WRIST_CAM_SERIAL, "WRIST")
    cam_main.start()
    cam_wrist.start()

    # ── Tactile (optional) ──
    fsr: Optional[FSRReader] = None
    if FSR_PORT.strip():
        try:
            fsr = FSRReader(
                FSR_PORT.strip(), FSR_BAUD, FSR_MODE, FSR_H,
                FSR_BASELINE_COUNT, FSR_KEEP_LAST_N, FSR_SCALE_DIV,
            )
        except Exception as e:
            log.warning(f"FSR init failed: {e}")

    # If FSR is enabled and FSR_WAIT_TIMEOUT > 0, wait until the baseline and a valid tactile window are ready.
    if fsr is not None and FSR_WAIT_TIMEOUT > 0:
        deadline = time.time() + FSR_WAIT_TIMEOUT
        log.info("Waiting for FSR baseline + valid tactile window (%.0fs timeout)...", FSR_WAIT_TIMEOUT)
        while time.time() < deadline:
            tac_l, tac_r = fsr.get_window(FSR_T)
            if tac_l is not None and tac_r is not None:
                log.info("FSR ready. Starting inference.")
                break
            time.sleep(0.5)
        else:
            log.warning("FSR did not become ready in time. Proceeding with zero tactile until ready.")

    grip = GripperState()
    dt = 1.0 / CTRL_HZ

    # Tactile cache: match run_loop behavior by using the last frame or zeros until the window is ready.
    last_tactile_left: Optional[np.ndarray] = None
    last_tactile_right: Optional[np.ndarray] = None
    zero_tac = np.zeros((FSR_T, FSR_H, FSR_W), dtype=np.float32)

    log.info("Running — press Ctrl+C to stop.")

    try:
        while True:
            # ── 1. Read observation ──
            code, angles = arm.get_servo_angle(is_radian=True)
            if code != 0:
                time.sleep(0.01)
                continue
            q6 = np.array(angles[:6], dtype=np.float32)

            g_bin = grip.update(read_gripper(arm), time.time())
            state = np.concatenate([q6, [g_bin]]).astype(np.float32)

            main_img = cam_main.read_rgb()
            wrist_img = cam_wrist.read_rgb()

            obs = {
                "observation/image": main_img,
                "observation/wrist_image": wrist_img,
                "observation/state": state,
                "prompt": PROMPT,
            }

            # Tactile: when use_tactile=True, the server expects tactile_left / tactile_right every request.
            # Prefer a real window, then the cached previous frame, then zeros.
            if fsr is not None:
                tac_l, tac_r = fsr.get_window(FSR_T)
            else:
                tac_l, tac_r = None, None
            if tac_l is not None and tac_r is not None:
                obs["observation/tactile_left"] = tac_l
                obs["observation/tactile_right"] = tac_r
                last_tactile_left = tac_l
                last_tactile_right = tac_r
            else:
                if last_tactile_left is not None and last_tactile_right is not None:
                    obs["observation/tactile_left"] = last_tactile_left.copy()
                    obs["observation/tactile_right"] = last_tactile_right.copy()
                else:
                    obs["observation/tactile_left"] = zero_tac
                    obs["observation/tactile_right"] = zero_tac

            # ── 2. Infer (blocks until server responds) ──
            t_infer = time.time()
            result = client.infer(obs)
            chunk = np.asarray(result["actions"], dtype=np.float32)
            log.info(f"Infer {(time.time() - t_infer) * 1000:.0f}ms  chunk={chunk.shape}")

            # ── 3. Execute chunk step-by-step ──
            n = min(len(chunk), OPEN_LOOP_HORIZON)
            for i in range(n):
                t_step = time.time()

                q_cmd = chunk[i, :6]
                g_cmd = float(np.clip(chunk[i, 6], 0, 1))

                arm.set_servo_angle(
                    angle=q_cmd.tolist() + [0.0],
                    speed=SERVO_SPEED,
                    mvacc=SERVO_MVACC,
                    wait=False,
                )
                send_gripper(arm, g_cmd)

                elapsed = time.time() - t_step
                if elapsed < dt:
                    time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        log.info("Ctrl+C — stopping")
    finally:
        try:
            arm.set_state(4)
        except Exception:
            pass
        try:
            arm.disconnect()
        except Exception:
            pass
        cam_main.stop()
        cam_wrist.stop()
        if fsr:
            fsr.stop()


if __name__ == "__main__":
    main()
