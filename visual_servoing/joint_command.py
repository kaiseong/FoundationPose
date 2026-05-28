#!/usr/bin/env python3
"""Send one right-arm joint command and print the resulting T5-frame EE pose."""

from __future__ import annotations

import argparse
import math

import numpy as np
import rby1_sdk as rby


DEFAULT_RIGHT_ARM_JOINT_RAD = (-0.0, 0.0, 0.0, -1.65, 0.0, 0.0, math.pi / 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a right-arm joint command and read Cartesian FK afterward.")
    parser.add_argument("--address", default="localhost:50051")
    parser.add_argument("--model", default="m")
    parser.add_argument("--power", default=".*")
    parser.add_argument("--servo", default=".*")
    parser.add_argument("--root-link", default="link_torso_5")
    parser.add_argument("--ee-link", default="ee_right")
    parser.add_argument("--priority", type=int, default=10)
    parser.add_argument("--minimum-time-s", type=float, default=1.0)
    parser.add_argument("--hold-time-s", type=float, default=1.0)
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--joint-rad",
        type=float,
        nargs=7,
        default=DEFAULT_RIGHT_ARM_JOINT_RAD,
        metavar=("J0", "J1", "J2", "J3", "J4", "J5", "J6"),
        help="Right-arm joint target in radians.",
    )
    target.add_argument(
        "--joint-deg",
        type=float,
        nargs=7,
        metavar=("J0", "J1", "J2", "J3", "J4", "J5", "J6"),
        help="Right-arm joint target in degrees.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    robot = rby.create_robot(args.address, args.model)
    try:
        connect_robot(robot, args)
        print_pose("before", compute_fk(robot, args.root_link, args.ee_link))

        joint_target = joint_target_rad(args)
        rc = rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(
                rby.BodyComponentBasedCommandBuilder().set_right_arm_command(
                    rby.JointPositionCommandBuilder()
                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(float(args.hold_time_s)))
                    .set_minimum_time(float(args.minimum_time_s))
                    .set_position(joint_target)
                )
            )
        )
        feedback = robot.send_command(rc, int(args.priority)).get()
        print(f"feedback={feedback}")
        print_pose("after", compute_fk(robot, args.root_link, args.ee_link))
    finally:
        if hasattr(robot, "disconnect"):
            robot.disconnect()
    return 0


def connect_robot(robot, args: argparse.Namespace) -> None:
    if not robot.connect():
        raise RuntimeError(f"Failed to connect to the robot: {args.address}")
    if not robot.is_power_on(args.power) and not robot.power_on(args.power):
        raise RuntimeError("Failed to power on")
    if not robot.is_servo_on(args.servo) and not robot.servo_on(args.servo):
        raise RuntimeError("Failed to servo on")
    if robot.get_control_manager_state().state in [
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ]:
        if not robot.reset_fault_control_manager():
            raise RuntimeError("Failed Control Manager Reset")
    if not robot.enable_control_manager():
        raise RuntimeError("Failed enable control manager")


def joint_target_rad(args: argparse.Namespace) -> np.ndarray:
    if args.joint_deg is not None:
        return np.deg2rad(np.asarray(args.joint_deg, dtype=np.float64))
    return np.asarray(args.joint_rad, dtype=np.float64)


def compute_fk(robot, root_link: str, ee_link: str) -> np.ndarray:
    robot_state = robot.get_state()
    dyn_model = robot.get_dynamics()
    dyn_state = dyn_model.make_state([root_link, ee_link], robot.model().robot_joint_names)
    dyn_state.set_q(robot_state.position)
    dyn_model.compute_forward_kinematics(dyn_state)
    return np.asarray(dyn_model.compute_transformation(dyn_state, 0, 1), dtype=np.float64)


def print_pose(label: str, root_t_ee: np.ndarray) -> None:
    xyz = root_t_ee[:3, 3]
    print(f"{label}_t5_xyz_m=({xyz[0]:.6f},{xyz[1]:.6f},{xyz[2]:.6f})")
    print(f"{label}_t5_T_ee=")
    print(np.array2string(root_t_ee, precision=6, suppress_small=True))


if __name__ == "__main__":
    raise SystemExit(main())
