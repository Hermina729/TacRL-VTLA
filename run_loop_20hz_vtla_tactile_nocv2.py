#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UF850 + OpenPI pi0.5 + VTLA(Tactile) 20Hz inference loop.

Changes from previous version (search "FIX:" for all changes):
  FIX:1 — Only add tactile keys to request when data is a real array (not None).
           Sending None causes TactilePreprocess to crash on np.asarray(None).
  FIX:2 — Removed redundant "observation.language_instruction" key (not used by UF850Inputs).
  FIX:3 — Added startup banner showing whether tactile is enabled.
"""

import os
import time
import threading
from collections import deque
import asyncio
from typing import Optional, Tuple, List
import numpy as np
try:
    from PIL import Image
except Exception:
    Image = None
import pyrealsense2 as rs
import websockets
import serial

from openpi_client import msgpack_numpy as client_mnp
from xarm.wrapper import XArmAPI
from collections import deque
_latency_log = deque(maxlen=30)   # Keep the most recent 30 inference latencies.

# ---------------- config ----------------
URI = os.environ.get("WS_URL", "ws://127.0.0.1:8000")
ROBOT_IP = os.environ.get("ROBOT_IP", "192.168.1.117")

MAIN_RS_SERIAL = os.environ.get("MAIN_RS_SERIAL", "216322072605")
WRIST_RS_SERIAL = os.environ.get("WRIST_RS_SERIAL", "148522073633")

IMG_SIZE = (224, 224)

RS_W = 640
RS_H = 480
RS_FPS = 30

CTRL_HZ = 50
CTRL_DT = 1.0 / CTRL_HZ

# Model horizon (how many steps you enqueue per inference)
K = 30
REFILL_WHEN_LEFT = 15
MAX_BUF = 30

# ACT: drop stale actions older than this (seconds)
ACTION_STALE_S = float(os.environ.get("ACTION_STALE_S", "0.8"))

# RTC / joint tracking
MAX_STEP_RAD = 0.05
EMA_ALPHA = 0.50
DEADZONE_RAD = 0.001


# ---------------- FSR / tactile (VTLA) ----------------
# If FSR_PORT is unset, tactile is disabled and the model will run image+state only.
FSR_PORT = os.environ.get("FSR_PORT", "")  # e.g. /dev/ttyACM0 or COM5
FSR_BAUD = int(os.environ.get("FSR_BAUD", "500000"))
FSR_MODE = int(os.environ.get("FSR_MODE", "1"))  # 0: single(16x16), 1: double(16x32)
FSR_H = int(os.environ.get("FSR_H", "16"))
FSR_W = int(os.environ.get("FSR_W", "16"))
FSR_T = int(os.environ.get("FSR_T", "5"))  # window length (must match training config tactile_T)
FSR_KEEP_LAST_N = int(os.environ.get("FSR_KEEP_LAST_N", "200"))
FSR_BASELINE_COUNT = int(os.environ.get("FSR_BASELINE_COUNT", "120"))
FSR_SCALE_DIV = float(os.environ.get("FSR_SCALE_DIV", "50.0"))

# ---------------- Gripper semantics ----------------
# Command semantic (your training): g_abs in [0,1], 1=open(850), 0=close(0)
GRIP_MAX = 850
GRIP_MIN = float(os.environ.get("GRIP_MIN", "45"))

# IMPORTANT: too large deadzone may kill small open signals
GRIP_DEADZONE = float(os.environ.get("GRIP_DEADZONE", "0.002"))

# STATE semantic (your training): g_state in {0,1} ; 1=fully open ; 0=closed/gripping/partial
OPEN_TH = float(os.environ.get("GRIP_OPEN_TH", "0.965")) * GRIP_MAX
CLOSE_TH = float(os.environ.get("GRIP_CLOSE_TH", "0.2")) * GRIP_MAX
POS_MAX_JUMP = float(os.environ.get("GRIP_POS_MAX_JUMP", "300"))  # reject sudden spikes
STATE_COOLDOWN_S = float(os.environ.get("GRIP_STATE_COOLDOWN_S", "0.3"))

# Gripper command rate-limit
GRIP_CMD_HZ = 10
GRIP_CMD_DT = 1.0 / GRIP_CMD_HZ
GRIP_CMD_MIN_STEP = 3.0

#PROMPT = os.environ.get("PROMPT", "grasp the handle and pull the drawer open")
PROMPT = os.environ.get("PROMPT", "grasp the test tube and insert it into the target position on the rack")

PRINT_HZ = 10
PRINT_DT = 1.0 / PRINT_HZ

CMD_PRINT_HZ = 1
CMD_PRINT_DT = 1.0 / CMD_PRINT_HZ


# ---------------- globals ----------------
stop_event = threading.Event()

# ACT buffer: each item is (ts, action7)
buf = deque(maxlen=MAX_BUF)
buf_lock = threading.Lock()


# ---------------- utils ----------------
def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def ema(prev: Optional[np.ndarray], new: np.ndarray, alpha: float) -> np.ndarray:
    return new if prev is None else alpha * new + (1 - alpha) * prev


def map_gripper_to_pos(g_norm_01: float) -> float:
    g = float(np.clip(g_norm_01, 0.0, 1.0))
    if g < GRIP_DEADZONE:
        g = 0.0
    pos = g * GRIP_MAX
    return float(np.clip(pos, GRIP_MIN, float(GRIP_MAX)))


def build_sdk_cmd7(q6: np.ndarray) -> List[float]:
    q6 = np.asarray(q6, dtype=np.float32).reshape(6)
    return q6.tolist() + [0.0]


def unpack_actions(out: dict) -> np.ndarray:
    return np.asarray(out["actions"])


# ---------------- Robust gripper binarization (STATE only) ----------------
_last_valid_pos: Optional[float] = None
_last_g_state: float = 0.0
_last_flip_t: float = 0.0


def _is_finite(x) -> bool:
    try:
        return bool(np.isfinite(x))
    except Exception:
        return False


def gripper_pos_to_bin_filtered(pos_read, now_t: float) -> Tuple[float, float]:
    global _last_valid_pos, _last_g_state, _last_flip_t

    pos: Optional[float] = None
    if pos_read is not None and _is_finite(pos_read):
        p = float(pos_read)
        if 0.0 <= p <= float(GRIP_MAX):
            pos = p

    if pos is not None and _last_valid_pos is not None:
        if abs(pos - _last_valid_pos) > POS_MAX_JUMP:
            pos = None

    if pos is None:
        pos = 0.0 if _last_valid_pos is None else float(_last_valid_pos)

    _last_valid_pos = pos

    g = float(_last_g_state)
    if (now_t - _last_flip_t) >= STATE_COOLDOWN_S:
        if g < 0.5:
            if pos >= OPEN_TH:
                g = 1.0
                _last_flip_t = now_t
        else:
            if pos <= CLOSE_TH:
                g = 0.0
                _last_flip_t = now_t

    _last_g_state = g
    return g, float(pos)


# ---------------- FSR Reader (threaded, low-latency) ----------------
class BaselineManager:
    def __init__(self, collect_count: int = 120):
        self.collect_count = int(collect_count)
        self.readings = []
        self.baseline = None

    def process(self, data_raw: np.ndarray) -> np.ndarray:
        if self.baseline is None:
            self.readings.append(data_raw)
            if len(self.readings) >= self.collect_count:
                self.baseline = np.mean(self.readings, axis=0)
                print("[FSR] ✓ Baseline computed.")
            return data_raw
        return data_raw - self.baseline


class FSRReader:
    def __init__(
        self,
        port: str,
        baudrate: int = 500000,
        mode: int = 1,
        size: int = 16,
        baseline_count: int = 120,
        keep_last_n: int = 200,
        scale_div: float = 50.0,
    ):
        self.port = port
        self.baudrate = int(baudrate)
        self.mode = int(mode)
        self.size = int(size)
        self.scale_div = float(scale_div)

        self.reorder_index = [8, 9, 10, 11, 12, 13, 14, 15, 7, 6, 5, 4, 3, 2, 1, 0]

        self.baseline1 = BaselineManager(baseline_count)
        self.baseline2 = BaselineManager(baseline_count)

        self._lock = threading.Lock()
        self._buf = deque(maxlen=keep_last_n)
        self._latest = None

        self._stop_event = threading.Event()
        self._thread = None
        self.ser = None
        self.running = False

    def start(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=3)
            time.sleep(1.0)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.running = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True, name="FSRSerialThread")
            self._thread.start()
            print(f"[FSR] ✓ started on {self.port} (baud={self.baudrate}, mode={self.mode})")
        except Exception as e:
            print(f"[FSR] ✗ start failed: {repr(e)}")
            self.running = False

    def stop(self):
        self.running = False
        self._stop_event.set()
        try:
            if self._thread:
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        print("[FSR] stopped.")

    def _run_loop(self):
        cmd = (self.mode << 6) | self.size
        cmd_bytes = cmd.to_bytes(1, "big")

        time_len = 4
        if self.mode == 0:
            frame_len = 2 * self.size * self.size
        else:
            frame_len = 4 * self.size * self.size
        data_len = frame_len + time_len

        while not self._stop_event.is_set():
            try:
                self.ser.write(cmd_bytes)
                d = self.ser.read(data_len)
                if len(d) != data_len:
                    continue

                frame_data = d[:-time_len]
                time_data = d[-time_len:]
                t_dev = int.from_bytes(time_data, "little")
                t_host = time.time()

                if self.mode == 0:
                    fsr_raw = np.frombuffer(frame_data, dtype=np.uint16).reshape(self.size, self.size)
                    fsr1_raw, fsr2_raw = fsr_raw, None
                else:
                    fsr_raw = np.frombuffer(frame_data, dtype=np.uint16).reshape(self.size, self.size * 2)
                    fsr1_raw = fsr_raw[:, : self.size]
                    fsr2_raw = fsr_raw[:, self.size :]

                fsr1 = fsr1_raw[self.reorder_index, :].astype(np.float32)
                fsr1 = self.baseline1.process(fsr1)
                fsr1 = np.clip(fsr1, 0.0, None) / self.scale_div

                fsr2 = None
                if fsr2_raw is not None:
                    fsr2 = fsr2_raw[self.reorder_index, :].astype(np.float32)
                    fsr2 = self.baseline2.process(fsr2)
                    fsr2 = np.clip(fsr2, 0.0, None) / self.scale_div

                valid = (self.baseline1.baseline is not None)
                if self.mode == 1:
                    valid = valid and (self.baseline2.baseline is not None)

                item = {"t_host": t_host, "t_dev": t_dev, "fsr1": fsr1, "fsr2": fsr2, "valid": valid}

                with self._lock:
                    self._latest = item
                    self._buf.append(item)
            except Exception as e:
                print(f"[FSR][WARN] {repr(e)}")
                time.sleep(0.05)

    def get_last_k(self, k: int):
        with self._lock:
            xs = list(self._buf)[-int(k):]
        return xs


# ---------------- RealSense ----------------
def rs_wait_device(serial: str, timeout_s: float = 8.0) -> bool:
    ctx = rs.context()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            for d in ctx.query_devices():
                if d.get_info(rs.camera_info.serial_number) == serial:
                    return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def rs_stop_safe(p):
    try:
        if p is not None:
            p.stop()
    except Exception:
        pass


class RSStream:
    def __init__(self, serial: str, name: str):
        self.serial = serial
        self.name = name
        self.pipeline = None
        self.backoff = 0.5

    def start(self):
        if not rs_wait_device(self.serial, timeout_s=8.0):
            raise RuntimeError(f"[{self.name}] Device not found: {self.serial}")
        p = rs.pipeline()
        c = rs.config()
        c.enable_device(self.serial)
        c.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
        p.start(c)
        for _ in range(10):
            try:
                p.poll_for_frames()
            except Exception:
                pass
        self.pipeline = p
        self.backoff = 0.5

    def restart(self, reason: str):
        print(f"[RS][{self.name}] restart due to: {reason}")
        rs_stop_safe(self.pipeline)
        self.pipeline = None
        time.sleep(self.backoff)
        self.backoff = min(self.backoff * 1.7, 5.0)
        self.start()

    def read_rgb(self, timeout_s: float = 0.6) -> np.ndarray:
        if self.pipeline is None:
            self.start()

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            frames = self.pipeline.poll_for_frames()
            if frames:
                cf = frames.get_color_frame()
                if cf:
                    img_bgr = np.asanyarray(cf.get_data())
                    img_rgb = img_bgr[..., ::-1]
                    if Image is not None:
                        img_rgb = np.asarray(
                            Image.fromarray(img_rgb).resize(IMG_SIZE, resample=Image.BILINEAR),
                            dtype=np.uint8,
                        )
                    else:
                        h, w = img_rgb.shape[:2]
                        th, tw = IMG_SIZE[1], IMG_SIZE[0]
                        ys = (np.linspace(0, h - 1, th)).astype(np.int32)
                        xs = (np.linspace(0, w - 1, tw)).astype(np.int32)
                        img_rgb = img_rgb[ys][:, xs].astype(np.uint8)
                    return np.ascontiguousarray(img_rgb, dtype=np.uint8)
            time.sleep(0.002)

        raise RuntimeError(f"[{self.name}] poll_for_frames timeout")

    def stop(self):
        rs_stop_safe(self.pipeline)
        self.pipeline = None


# ---------------- Gripper IO (safe) ----------------
def read_gripper_pos_safe(arm: XArmAPI) -> Tuple[bool, Optional[float]]:
    if hasattr(arm, "get_gripper_position"):
        try:
            ret = arm.get_gripper_position()
            if isinstance(ret, (list, tuple)) and len(ret) >= 2 and ret[0] == 0:
                return True, float(ret[1])
        except Exception:
            pass
    for name in ["get_gripper_pos", "get_gripper_state", "get_gripper_status"]:
        if hasattr(arm, name):
            try:
                ret = getattr(arm, name)()
                if isinstance(ret, (list, tuple)) and len(ret) >= 2 and ret[0] == 0:
                    return True, float(ret[1])
            except Exception:
                pass
    return False, None


def set_gripper_pos_safe(arm: XArmAPI, pos: float) -> bool:
    pos = float(np.clip(pos, GRIP_MIN, float(GRIP_MAX)))
    if hasattr(arm, "set_gripper_position"):
        try:
            arm.set_gripper_position(pos, wait=False)
            return True
        except Exception:
            return False
    if hasattr(arm, "set_gripper_pos"):
        try:
            arm.set_gripper_pos(pos, wait=False)
            return True
        except Exception:
            return False
    return False


# ---------------- inference (async ws) ----------------

async def inference_loop_async(get_state_fn):
    packer = client_mnp.Packer()
    rs_main = RSStream(MAIN_RS_SERIAL, "MAIN")
    rs_wrist = RSStream(WRIST_RS_SERIAL, "WRIST")

    # FSR reader (optional)
    fsr_reader = None
    last_tactile_left = None
    last_tactile_right = None
    if FSR_PORT.strip():
        try:
            fsr_reader = FSRReader(
                port=FSR_PORT.strip(),
                baudrate=FSR_BAUD,
                mode=FSR_MODE,
                size=FSR_H,
                baseline_count=FSR_BASELINE_COUNT,
                keep_last_n=FSR_KEEP_LAST_N,
                scale_div=FSR_SCALE_DIV,
            )
            fsr_reader.start()
        except Exception as e:
            print(f"[FSR][WARN] init failed: {repr(e)}")
            fsr_reader = None

    ws = None

    async def connect():
        nonlocal ws
        ws = await websockets.connect(
            URI,
            open_timeout=5,
            max_size=None,
            ping_interval=30,
            ping_timeout=30,
        )
        try:
            _ = await ws.recv()
        except Exception:
            pass

    async def close_ws():
        nonlocal ws
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
            ws = None

    warm = 0
    try:
        await connect()

        while not stop_event.is_set():
            with buf_lock:
                left = len(buf)
            if left > REFILL_WHEN_LEFT:
                await asyncio.sleep(0.005)
                continue

            try:
                state_now = get_state_fn().astype(np.float32)

                try:
                    main_img = rs_main.read_rgb(timeout_s=0.6)
                except Exception as e:
                    print(f"[RS][MAIN][WARN] {repr(e)} -> restart")
                    rs_main.restart(str(e))
                    main_img = rs_main.read_rgb(timeout_s=0.6)

                try:
                    wrist_img = rs_wrist.read_rgb(timeout_s=0.6)
                except Exception as e:
                    print(f"[RS][WRIST][WARN] {repr(e)} -> restart")
                    rs_wrist.restart(str(e))
                    wrist_img = rs_wrist.read_rgb(timeout_s=0.6)

                # ---- tactile (optional) ----
                tactile_left = None
                tactile_right = None
                if fsr_reader is not None:
                    xs = [x for x in fsr_reader.get_last_k(max(2 * FSR_T, 10)) if x.get("valid", False)]
                    xs = xs[-FSR_T:] if len(xs) >= FSR_T else xs

                    if len(xs) == FSR_T:
                        tactile_left = np.stack([x["fsr1"] for x in xs], axis=0).astype(np.float32)
                        if FSR_MODE == 1 and all(x.get("fsr2") is not None for x in xs):
                            tactile_right = np.stack([x["fsr2"] for x in xs], axis=0).astype(np.float32)
                        else:
                            tactile_right = np.zeros((FSR_T, FSR_H, FSR_W), dtype=np.float32)
                    else:
                        tactile_left = last_tactile_left.copy() if last_tactile_left is not None else np.zeros((FSR_T, FSR_H, FSR_W), dtype=np.float32)
                        tactile_right = last_tactile_right.copy() if last_tactile_right is not None else np.zeros((FSR_T, FSR_H, FSR_W), dtype=np.float32)

                    # update cache
                    last_tactile_left = tactile_left
                    last_tactile_right = tactile_right

                # ---- build request ----
                # FIX:1 — Only include tactile keys when we have real arrays.
                #          Sending None values causes TactilePreprocess to crash
                #          (np.asarray(None) → 0-d object array → ndim<3 → ValueError).
                req = {
                    "observation/image": main_img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state_now,
                    "prompt": PROMPT,
                    # FIX:2 — Removed "observation.language_instruction" (not used by UF850Inputs)
                }

                if tactile_left is not None and tactile_right is not None:
                    req["observation/tactile_left"] = tactile_left
                    req["observation/tactile_right"] = tactile_right

                payload = packer.pack(req)

                if ws is None:
                    await connect()

                # ======== Timing starts ========
                t_send = time.time()
                try:
                    await ws.send(payload)
                    resp = await ws.recv()
                except Exception:
                    await close_ws()
                    await connect()
                    t_send = time.time()   # Reset timing after reconnecting.
                    await ws.send(payload)
                    resp = await ws.recv()
                t_recv = time.time()

                infer_latency_ms = (t_recv - t_send) * 1000
                # ======== Timing ends ========

                if isinstance(resp, str):
                    print("[INFER][SERVER_TEXT]\n", resp)
                    await asyncio.sleep(0.2)
                    continue

                out = client_mnp.unpackb(resp)
                actions = np.asarray(unpack_actions(out), dtype=np.float32)

                # ACT: timestamp
                ts0 = time.time()
                added = 0
                with buf_lock:
                    space = MAX_BUF - len(buf)
                    n = int(min(actions.shape[0], K, space))
                    for i in range(n):
                        buf.append((ts0 + i * CTRL_DT, actions[i].copy()))
                    added = n

                warm += 1
                tag = "WARMUP" if warm <= 2 else "RUN"
                with buf_lock:
                    bl = len(buf)

                # ======== Print latency, including breakdown fields ========
                print(f"[INFER][{tag}] latency={infer_latency_ms:.0f}ms  +{added} buf={bl}")

            except Exception as e:
                print(f"[INFER][WARN] {repr(e)}")
                await asyncio.sleep(0.2)

    finally:
        await close_ws()
        rs_main.stop()
        rs_wrist.stop()
        if fsr_reader is not None:
            fsr_reader.stop()


def inference_worker(get_state_fn):
    asyncio.run(inference_loop_async(get_state_fn))


# ---------------- main control ----------------
def main():
    # FIX:3 — Print tactile config at startup
    print("=" * 60)
    print("  UF850 + OpenPI pi0.5 + VTLA Tactile Run Loop")
    print("=" * 60)
    if FSR_PORT.strip():
        print(f"  Tactile: ENABLED  port={FSR_PORT}  mode={FSR_MODE}  T={FSR_T}  H={FSR_H}  W={FSR_W}")
    else:
        print("  Tactile: DISABLED (set FSR_PORT to enable)")
    print(f"  Server:  {URI}")
    print(f"  Robot:   {ROBOT_IP}")
    print(f"  Prompt:  {PROMPT}")
    print("=" * 60)

    arm = XArmAPI(ROBOT_IP, is_radian=True)
    arm.connect()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)

    print("[INFO] has set_gripper_position:", hasattr(arm, "set_gripper_position"),
          "has set_gripper_pos:", hasattr(arm, "set_gripper_pos"))
    print("[INFO] has get_gripper_position:", hasattr(arm, "get_gripper_position"))
    print(f"[INFO][GRIP] OPEN_TH={OPEN_TH:.1f} CLOSE_TH={CLOSE_TH:.1f} "
          f"MAX_JUMP={POS_MAX_JUMP:.1f} COOLDOWN={STATE_COOLDOWN_S:.3f}s DEADZONE={GRIP_DEADZONE}")

    # -------- shared state for inference (7D: q6 + g_bin) --------
    state_lock = threading.Lock()

    code, q_full = arm.get_servo_angle(is_radian=True)
    if code != 0:
        raise RuntimeError(f"get_servo_angle failed: {code}")
    if len(q_full) < 6:
        raise RuntimeError(f"Unexpected get_servo_angle length: {len(q_full)}")
    q6 = np.array(q_full[:6], dtype=np.float32)

    ok_g, pos = read_gripper_pos_safe(arm)
    now0 = time.time()
    pos_read0 = float(pos) if (ok_g and pos is not None) else None
    g_state0, grip_pos0 = gripper_pos_to_bin_filtered(pos_read0, now0)

    state_for_infer = np.concatenate([q6, [g_state0]]).astype(np.float32)

    def get_state_copy():
        with state_lock:
            return state_for_infer.copy()

    # -------- desired targets (ABS) --------
    target_lock = threading.Lock()
    desired_q_abs = q6.copy()
    desired_g_abs = 1.0

    # -------- feedback --------
    fb_lock = threading.Lock()
    q_fb = q6.copy()
    grip_pos_fb = float(grip_pos0)
    g_state_fb = float(g_state0)

    # start inference thread
    th = threading.Thread(target=inference_worker, args=(get_state_copy,), daemon=True)
    th.start()

    # -------- RTC sender thread (fixed-rate output) --------
    prev_step: Optional[np.ndarray] = None

    def sender_worker():
        nonlocal prev_step, desired_g_abs

        send_hz = CTRL_HZ
        dt = 1.0 / send_hz
        last_cmd_print = 0.0

        SERVO_SPEED = 120
        SERVO_MVACC = 2000

        last_grip_send_t = 0.0
        last_grip_pos_cmd: Optional[float] = None

        while not stop_event.is_set():
            t0 = time.time()

            with target_lock:
                q_tgt = desired_q_abs.copy()
                g_cmd = float(desired_g_abs)

            with fb_lock:
                q_now = q_fb.copy()

            err = (q_tgt - q_now).astype(np.float32)
            step = clamp(err, -MAX_STEP_RAD, MAX_STEP_RAD)
            step = np.where(np.abs(step) < DEADZONE_RAD, 0.0, step)
            step = ema(prev_step, step, EMA_ALPHA)
            prev_step = step
            q_send = (q_now + step).astype(np.float32)

            cmd7 = build_sdk_cmd7(q_send)
            try:
                arm.set_servo_angle(angle=cmd7, speed=SERVO_SPEED, mvacc=SERVO_MVACC, wait=False)
            except Exception as e:
                print(f"[SEND_JOINT][WARN] {repr(e)}")

            now = time.time()
            if now - last_grip_send_t >= GRIP_CMD_DT:
                g_pos = map_gripper_to_pos(g_cmd)
                if (last_grip_pos_cmd is None) or (abs(g_pos - last_grip_pos_cmd) >= GRIP_CMD_MIN_STEP):
                    ok = set_gripper_pos_safe(arm, g_pos)
                    if ok:
                        last_grip_pos_cmd = g_pos
                    last_grip_send_t = now

            if time.time() - last_cmd_print >= CMD_PRINT_DT:
                print(
                    "[CMD]\n"
                    f"  q_tgt6={q_tgt.tolist()}\n"
                    f"  q_now6={q_now.tolist()}\n"
                    f"  step6 ={step.tolist()}\n"
                    f"  cmd7  ={cmd7}\n"
                    f"  g_cmd_abs={g_cmd:.3f} -> g_cmd_pos={map_gripper_to_pos(g_cmd):.1f}\n"
                )
                last_cmd_print = time.time()

            sleep_t = dt - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    sender_th = threading.Thread(target=sender_worker, daemon=True)
    sender_th.start()

    # -------- main control loop (ACT consumer + feedback) --------
    last_stat = time.time()
    ticks = 0
    last_read = 0.0
    FB_HZ = 20
    FB_DT = 1.0 / FB_HZ
    last_print = 0.0

    try:
        while True:
            t0 = time.time()
            now = time.time()

            if now - last_read >= FB_DT:
                code, q_now_full = arm.get_servo_angle(is_radian=True)
                if code == 0 and len(q_now_full) >= 6:
                    q_now6 = np.array(q_now_full[:6], dtype=np.float32)

                    ok_g, pos = read_gripper_pos_safe(arm)
                    pos_read = float(pos) if (ok_g and pos is not None) else None

                    g_state, grip_pos = gripper_pos_to_bin_filtered(pos_read, now)

                    with fb_lock:
                        q_fb[:] = q_now6
                        grip_pos_fb = float(grip_pos)
                        g_state_fb = float(g_state)

                    with state_lock:
                        state_for_infer[:] = np.concatenate([q_now6, [g_state]]).astype(np.float32)

                last_read = now

            item = None
            with buf_lock:
                if buf:
                    item = buf.popleft()

            if item is not None:
                ts_action, a = item
                age = time.time() - ts_action
                if age <= ACTION_STALE_S:
                    a = np.asarray(a, dtype=np.float32)
                    q_abs = a[:6]
                    g_abs = float(a[6])
                    

                    if not np.isfinite(g_abs):
                        g_abs = 1.0
                    g_abs = float(np.clip(g_abs, 0.0, 1.0))
                    print('q_abs')
                    print(q_abs)

                    with target_lock:
                        desired_q_abs[:] = q_abs
                        desired_g_abs = g_abs

                    if time.time() - last_print >= PRINT_DT:
                        with fb_lock:
                            q_now6_print = q_fb.copy()
                            grip_pos_print = float(grip_pos_fb)
                            g_state_print = float(g_state_fb)

                        state_now7 = np.concatenate([q_now6_print, [g_state_print]]).astype(np.float32)
                        action7 = np.concatenate([q_abs, [g_abs]]).astype(np.float32)

                        print(
                            "[SANITY7]\n"
                            f"  state_now7(q6+g_bin) = {state_now7.tolist()}  (grip_pos_filt={grip_pos_print:.1f})\n"
                            f"  action7(q_abs6+g_abs)= {action7.tolist()}  (g_cmd_pos={map_gripper_to_pos(g_abs):.1f})\n"
                            f"  action_age={age*1000:.0f}ms\n"
                        )
                        last_print = time.time()

            ticks += 1
            if time.time() - last_stat >= 1.0:
                hz = ticks / (time.time() - last_stat)
                with buf_lock:
                    blen = len(buf)
                with fb_lock:
                    g_state_show = float(g_state_fb)
                    grip_pos_show = float(grip_pos_fb)
                with target_lock:
                    g_cmd_show = float(desired_g_abs)
                print(f"[CTRL] hz={hz:.1f} buf={blen} grip_pos_filt={grip_pos_show:.1f} g_state={g_state_show:.0f} g_cmd_abs={g_cmd_show:.2f}")
                ticks = 0
                last_stat = time.time()

            sleep_t = CTRL_DT - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C -> stopping")
    finally:
        stop_event.set()
        try:
            arm.set_state(4)
        except Exception:
            pass
        try:
            arm.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
