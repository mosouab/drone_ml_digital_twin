#!/usr/bin/env python3
import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from tello_msgs.srv import TelloAction


def quat_to_euler_xyz(qx, qy, qz, qw):
    """Quaternion -> (roll,pitch,yaw) in radians."""
    # roll
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    # yaw
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return float(roll), float(pitch), float(yaw)


@dataclass
class Sample:
    t: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    vx: float
    vy: float
    vz: float
    ux: float
    uy: float
    uz: float
    uyaw: float


class DatasetNode(Node):
    def __init__(self, ns: str, model_name: str):
        super().__init__("tello_dataset_gen")

        self.ns = ns.strip("/")
        self.model_name = model_name

        self.cmd_vel_topic = f"/{self.ns}/cmd_vel"
        self.tello_action_srv = f"/{self.ns}/tello_action"
        self.model_states_topic = "/gazebo/model_states"

        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.sub_states = self.create_subscription(ModelStates, self.model_states_topic, self._on_model_states, 10)
        self.cli = self.create_client(TelloAction, self.tello_action_srv)

        self.last_cmd = (0.0, 0.0, 0.0, 0.0)
        self.samples = []

        # last known state from gazebo
        self._have_state = False
        self._x = self._y = self._z = 0.0
        self._roll = self._pitch = self._yaw = 0.0
        self._vx = self._vy = self._vz = 0.0

        # index caching
        self._idx = None

    def _on_model_states(self, msg: ModelStates):
        # find model index
        if self._idx is None:
            if self.model_name in msg.name:
                self._idx = msg.name.index(self.model_name)
            else:
                # fallback: first model containing tello/drone
                for i, n in enumerate(msg.name):
                    nn = n.lower()
                    if "tello" in nn or "drone" in nn:
                        self._idx = i
                        self.get_logger().warn(f"Model '{self.model_name}' not found. Using '{msg.name[i]}' instead.")
                        break
        if self._idx is None or self._idx >= len(msg.name):
            return

        pose = msg.pose[self._idx]
        twist = msg.twist[self._idx]

        self._x = float(pose.position.x)
        self._y = float(pose.position.y)
        self._z = float(pose.position.z)

        q = pose.orientation
        self._roll, self._pitch, self._yaw = quat_to_euler_xyz(q.x, q.y, q.z, q.w)

        self._vx = float(twist.linear.x)
        self._vy = float(twist.linear.y)
        self._vz = float(twist.linear.z)

        self._have_state = True

    def wait_service(self, timeout_sec=8.0) -> bool:
        t0 = time.time()
        while rclpy.ok() and not self.cli.wait_for_service(timeout_sec=0.5):
            if time.time() - t0 > timeout_sec:
                return False
        return True

    def call_action(self, cmd: str, timeout_sec=8.0) -> int:
        """Return rc from service, or -1 on failure."""
        req = TelloAction.Request()
        req.cmd = cmd
        fut = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout_sec)
        if not fut.done() or fut.result() is None:
            return -1
        return int(fut.result().rc)

    def publish_cmd(self, ux, uy, uz, uyaw):
        msg = Twist()
        msg.linear.x = float(ux)
        msg.linear.y = float(uy)
        msg.linear.z = float(uz)
        msg.angular.z = float(uyaw)
        self.pub_cmd.publish(msg)
        self.last_cmd = (float(ux), float(uy), float(uz), float(uyaw))

    def record_sample(self, t_now: float):
        if not self._have_state:
            return
        ux, uy, uz, uyaw = self.last_cmd
        self.samples.append(Sample(
            t=float(t_now),
            x=self._x, y=self._y, z=self._z,
            roll=self._roll, pitch=self._pitch, yaw=self._yaw,
            vx=self._vx, vy=self._vy, vz=self._vz,
            ux=ux, uy=uy, uz=uz, uyaw=uyaw
        ))


def scenario_cmd(t, name="hover"):
    if name == "hover":
        return 0.0, 0.0, 0.0, 0.0

    if name == "circle_xy":
        w = 0.25
        return 0.15*np.cos(w*t), 0.15*np.sin(w*t), 0.0, 0.2

    if name == "helix":
        w = 0.25
        uz = 0.08*np.sin(0.18*t)
        return 0.15*np.cos(w*t), 0.15*np.sin(w*t), uz, 0.2

    if name == "ramps":
        T = 8.0
        phase = (t % (4*T))
        if phase < T:
            return 0.2, 0.0, 0.0, 0.0
        if phase < 2*T:
            return 0.0, 0.2, 0.0, 0.0
        if phase < 3*T:
            return 0.0, 0.0, 0.12, 0.0
        return 0.0, 0.0, 0.0, 0.3

    if name == "random_smooth":
        return (
            0.10*np.sin(0.37*t) + 0.05*np.sin(0.11*t),
            0.10*np.sin(0.29*t) + 0.05*np.sin(0.07*t),
            0.08*np.sin(0.19*t),
            0.25*np.sin(0.13*t)
        )

    if name == "mixed":
        block = int((t // 30) % 4)
        if block == 0:
            return scenario_cmd(t, "hover")
        if block == 1:
            return scenario_cmd(t, "circle_xy")
        if block == 2:
            return scenario_cmd(t, "helix")
        return scenario_cmd(t, "ramps")

    return 0.0, 0.0, 0.0, 0.0


def build_datasets(df: pd.DataFrame, outdir: str, seq_len: int = 20):
    state_cols = ["x","y","z","roll","pitch","yaw","vx","vy","vz"]
    u_cols = ["ux","uy","uz","uyaw"]
    df = df.sort_values("t").reset_index(drop=True)

    # raw already exists; build derived files only if enough samples
    if len(df) < 2:
        return

    # transitions.csv
    trans = np.concatenate([
        df[state_cols].iloc[:-1].to_numpy(),
        df[u_cols].iloc[:-1].to_numpy(),
        df[state_cols].iloc[1:].to_numpy()
    ], axis=1)
    trans_cols = [f"s_{c}" for c in state_cols] + [f"u_{c}" for c in u_cols] + [f"sp1_{c}" for c in state_cols]
    trans_df = pd.DataFrame(trans, columns=trans_cols)
    trans_df.insert(0, "t", df["t"].iloc[:-1].to_numpy())
    trans_df.to_csv(os.path.join(outdir, "transitions.csv"), index=False)

    # sequences.npz
    if len(df) > seq_len + 1:
        S = df[state_cols].to_numpy()
        U = df[u_cols].to_numpy()
        t_all = df["t"].to_numpy()

        Xs, Xu, Ys, Ts = [], [], [], []
        for i in range(0, len(df) - seq_len - 1):
            Xs.append(S[i:i+seq_len])
            Xu.append(U[i:i+seq_len])
            Ys.append(S[i+seq_len])
            Ts.append(t_all[i+seq_len])

        np.savez_compressed(
            os.path.join(outdir, "sequences.npz"),
            X_state=np.asarray(Xs, dtype=np.float32),
            X_cmd=np.asarray(Xu, dtype=np.float32),
            Y_next=np.asarray(Ys, dtype=np.float32),
            t=np.asarray(Ts, dtype=np.float64),
            state_cols=np.array(state_cols),
            cmd_cols=np.array(u_cols),
        )

    # ode.csv
    dt = np.diff(df["t"].to_numpy())
    S = df[state_cols].to_numpy()
    dS = (S[1:] - S[:-1]) / dt[:, None]
    ode = np.concatenate([
        df["t"].iloc[:-1].to_numpy()[:,None],
        df[state_cols].iloc[:-1].to_numpy(),
        df[u_cols].iloc[:-1].to_numpy(),
        dt[:,None],
        dS
    ], axis=1)
    ode_cols = ["t"] + [f"s_{c}" for c in state_cols] + [f"u_{c}" for c in u_cols] + ["dt"] + [f"ds_{c}" for c in state_cols]
    pd.DataFrame(ode, columns=ode_cols).to_csv(os.path.join(outdir, "ode.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="drone1")
    ap.add_argument("--model-name", default="tello_1", help="Gazebo model name (seen in /gazebo/model_states)")
    ap.add_argument("--scenario", default="mixed",
                    choices=["hover","circle_xy","helix","ramps","random_smooth","mixed"])
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--out-root", default=os.path.expanduser("~/tello_datasets"))
    ap.add_argument("--no-takeoff", action="store_true")
    args = ap.parse_args()

    rclpy.init()
    node = DatasetNode(ns=args.ns, model_name=args.model_name)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.out_root, f"run_{ts}_{args.scenario}")
    os.makedirs(outdir, exist_ok=True)

    meta = {
        "ns": args.ns,
        "model_name": args.model_name,
        "scenario": args.scenario,
        "minutes": args.minutes,
        "rate": args.rate,
        "seq_len": args.seq_len,
        "topics": {
            "cmd_vel": f"/{args.ns}/cmd_vel",
            "tello_action": f"/{args.ns}/tello_action",
            "model_states": "/gazebo/model_states"
        }
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Wait for model_states to start arriving
    t_wait = time.time()
    while rclpy.ok() and (not node._have_state) and (time.time() - t_wait < 5.0):
        rclpy.spin_once(node, timeout_sec=0.1)

    if not node._have_state:
        node.get_logger().error("No /gazebo/model_states received. Check gazebo_ros_state plugin + topic name.")
        node.destroy_node()
        rclpy.shutdown()
        return

    # takeoff
    if node.wait_service(timeout_sec=8.0) and (not args.no_takeoff):
        node.get_logger().info("Calling takeoff...")
        rc = node.call_action("takeoff", timeout_sec=8.0)
        node.get_logger().info(f"takeoff rc={rc}")
        time.sleep(1.0)
    else:
        node.get_logger().warn("tello_action not ready OR --no-takeoff used. Continuing without takeoff.")

    T = args.minutes * 60.0
    dt = 1.0 / max(1e-3, args.rate)
    t0 = time.time()
    next_tick = t0

    node.get_logger().info(f"Recording {T:.1f}s at {args.rate:.1f} Hz -> {outdir}")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)

            now = time.time()
            if now < next_tick:
                time.sleep(max(0.0, next_tick - now))
            next_tick += dt

            t = time.time() - t0
            if t > T:
                break

            ux, uy, uz, uyaw = scenario_cmd(t, args.scenario)
            node.publish_cmd(ux, uy, uz, uyaw)
            node.record_sample(t_now=t)

    finally:
        node.publish_cmd(0.0, 0.0, 0.0, 0.0)
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.wait_service(timeout_sec=1.0):
            node.get_logger().info("Calling land...")
            rc = node.call_action("land", timeout_sec=8.0)
            node.get_logger().info(f"land rc={rc}")

    df = pd.DataFrame([s.__dict__ for s in node.samples])
    df.to_csv(os.path.join(outdir, "raw.csv"), index=False)
    build_datasets(df, outdir, seq_len=args.seq_len)

    node.get_logger().info(f"Done. Samples={len(df)} | Files in {outdir}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

