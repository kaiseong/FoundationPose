#!/usr/bin/env python3
"""Keyboard-driven T5 Cartesian motion test for the right arm.

This diagnostic bypasses camera perception and sends small Cartesian
translation steps directly in the configured control-root frame.
"""

from __future__ import annotations

import argparse
import curses
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

_ADDED_PACKAGE_PARENT: str | None = None
if __package__ in (None, ""):
    _PACKAGE_PARENT = str(Path(__file__).resolve().parents[1])
    if _PACKAGE_PARENT not in sys.path:
        sys.path.insert(0, _PACKAGE_PARENT)
        _ADDED_PACKAGE_PARENT = _PACKAGE_PARENT

from visual_servoing.visual_servo_client import (  # noqa: E402
    DEFAULT_RIGHT_ARM_EE_LINK,
    ROBOT_MODEL,
    RIGHT_ARM_CONTROL_ROOT_LINK,
    RobotContext,
    format_xyz_m,
    make_transform_from_xyz_rpy,
)

if _ADDED_PACKAGE_PARENT is not None:
    sys.path.remove(_ADDED_PACKAGE_PARENT)


AXIS_CHOICES = ("x", "y", "z", "-x", "-y", "-z")
KEY_PRESET_CHOICES = ("custom", "robot-plus-x-left-y-up-z", "robot-minus-x-left-y-up-z")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send small T5-frame Cartesian keyboard steps to the right-arm EE."
    )
    parser.add_argument("--address", help="Robot address; required with --execute.")
    parser.add_argument("--model", default=ROBOT_MODEL)
    parser.add_argument("--power", default=".*")
    parser.add_argument("--servo", default=".*")
    parser.add_argument("--control-root-link", default=RIGHT_ARM_CONTROL_ROOT_LINK)
    parser.add_argument("--ee-link", default=DEFAULT_RIGHT_ARM_EE_LINK)
    parser.add_argument(
        "--execute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Send commands to the real robot. Default is dry-run.",
    )
    parser.add_argument(
        "--move-to-ready-on-connect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Move the right arm to the visual-servo ready pose before keyboard control.",
    )
    parser.add_argument("--step-m", type=float, default=0.01, help="Translation step per key press.")
    parser.add_argument(
        "--key-preset",
        choices=KEY_PRESET_CHOICES,
        default="custom",
        help=(
            "Keyboard mapping preset. robot-plus-x-left-y-up-z maps Left/Right to +/-y, "
            "Up/Down to +/-z, and f/b to +/-x. robot-minus-x-left-y-up-z is the legacy "
            "inverted-x preset."
        ),
    )
    parser.add_argument(
        "--horizontal-axis",
        choices=AXIS_CHOICES,
        default="x",
        help="Custom mode axis controlled by Left/Right arrows. Right arrow applies the positive direction.",
    )
    parser.add_argument(
        "--vertical-axis",
        choices=AXIS_CHOICES,
        default="z",
        help="Custom mode axis controlled by Up/Down arrows. Up arrow applies the positive direction.",
    )
    parser.add_argument(
        "--current-ee-pose",
        type=float,
        nargs=6,
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Dry-run initial EE pose in T5 frame: meters and degrees.",
    )
    parser.add_argument("--command-min-time-s", type=float, default=0.25)
    parser.add_argument("--command-hold-time-s", type=float, default=0.5)
    parser.add_argument("--command-timeout-s", type=float, default=2.0)
    parser.add_argument("--command-priority", type=int, default=10)
    parser.add_argument("--control-ready-timeout-ms", type=int, default=1000)
    parser.add_argument("--ready-min-time-s", type=float, default=3.0)
    parser.add_argument("--ready-hold-time-s", type=float, default=4.0)
    parser.add_argument("--linear-limit", type=float, default=0.2)
    parser.add_argument("--angular-limit", type=float, default=math.pi / 4.0)
    parser.add_argument("--linear-gain", type=float, default=50.0)
    parser.add_argument("--angular-gain", type=float, default=math.pi * 20.0)
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.execute and not args.address:
        raise SystemExit("--address is required with --execute")
    if float(args.step_m) <= 0.0:
        raise SystemExit("--step-m must be positive")


def axis_delta(axis_name: str, amount_m: float) -> np.ndarray:
    sign = -1.0 if axis_name.startswith("-") else 1.0
    axis = axis_name[-1]
    index = {"x": 0, "y": 1, "z": 2}[axis]
    delta = np.zeros(3, dtype=np.float64)
    delta[index] = sign * float(amount_m)
    return delta


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    robot_context = RobotContext.connect(args) if args.execute else RobotContext.dry_run(args)
    try:
        return curses.wrapper(run_keyboard_loop, args, robot_context)
    finally:
        robot_context.close()


def run_keyboard_loop(stdscr: Any, args: argparse.Namespace, robot_context: RobotContext) -> int:
    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.keypad(True)

    current_t5_T_ee = robot_context.current_ee_pose()
    target_t5_T_ee = current_t5_T_ee.copy()
    reference_t5_R_ee = current_t5_T_ee[:3, :3].copy()
    last_feedback = "ready"

    horizontal_delta = axis_delta(args.horizontal_axis, args.step_m)
    vertical_delta = axis_delta(args.vertical_axis, args.step_m)
    key_map = make_key_map(args, horizontal_delta=horizontal_delta, vertical_delta=vertical_delta)

    while True:
        draw_screen(stdscr, args, robot_context, current_t5_T_ee, target_t5_T_ee, last_feedback)
        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):
            return 0
        if key in (ord("r"), ord("R")):
            current_t5_T_ee = robot_context.current_ee_pose()
            target_t5_T_ee = current_t5_T_ee.copy()
            reference_t5_R_ee = current_t5_T_ee[:3, :3].copy()
            last_feedback = "reset target to current FK"
            continue

        delta = key_map.get(key)
        if delta is None:
            last_feedback = "ignored key"
            continue

        target_t5_T_ee[:3, :3] = reference_t5_R_ee
        target_t5_T_ee[:3, 3] = target_t5_T_ee[:3, 3] + delta
        last_feedback = send_or_preview(robot_context, target_t5_T_ee)
        if robot_context.execute:
            current_t5_T_ee = robot_context.current_ee_pose()
        else:
            current_t5_T_ee = target_t5_T_ee.copy()


def make_key_map(
    args: argparse.Namespace,
    *,
    horizontal_delta: np.ndarray,
    vertical_delta: np.ndarray,
) -> dict[int, np.ndarray]:
    if args.key_preset in {"robot-plus-x-left-y-up-z", "robot-minus-x-left-y-up-z"}:
        robot_toward = axis_delta("x", args.step_m)
        if args.key_preset == "robot-minus-x-left-y-up-z":
            robot_toward = -robot_toward
        robot_left = axis_delta("y", args.step_m)
        up = axis_delta("z", args.step_m)
        return {
            curses.KEY_LEFT: robot_left,
            curses.KEY_RIGHT: -robot_left,
            curses.KEY_UP: up,
            curses.KEY_DOWN: -up,
            ord("f"): robot_toward,
            ord("F"): robot_toward,
            curses.KEY_NPAGE: robot_toward,
            ord("b"): -robot_toward,
            ord("B"): -robot_toward,
            curses.KEY_PPAGE: -robot_toward,
        }
    return {
        curses.KEY_RIGHT: horizontal_delta,
        curses.KEY_LEFT: -horizontal_delta,
        curses.KEY_UP: vertical_delta,
        curses.KEY_DOWN: -vertical_delta,
    }


def send_or_preview(robot_context: RobotContext, target_t5_T_ee: np.ndarray) -> str:
    if not robot_context.execute:
        return "dry-run target updated"
    feedback = robot_context.send_right_arm_cartesian(target_t5_T_ee)
    return "sent " + " ".join(f"{key}={value}" for key, value in feedback.items())


def draw_screen(
    stdscr: Any,
    args: argparse.Namespace,
    robot_context: RobotContext,
    current_t5_T_ee: np.ndarray,
    target_t5_T_ee: np.ndarray,
    last_feedback: str,
) -> None:
    stdscr.erase()
    mode = "EXECUTE" if robot_context.execute else "DRY-RUN"
    current_xyz = tuple(float(value) for value in current_t5_T_ee[:3, 3])
    target_xyz = tuple(float(value) for value in target_t5_T_ee[:3, 3])
    lines = [
        "Right-arm T5 Cartesian keyboard test",
        f"mode={mode} root={args.control_root_link} ee={args.ee_link} step_m={float(args.step_m):.3f}",
        key_help(args),
        "q or Esc: quit    r: reset target to current FK",
        "",
        f"current_t5_xyz_m={format_xyz_m(current_xyz)}",
        f"target_t5_xyz_m ={format_xyz_m(target_xyz)}",
        f"last={last_feedback}",
    ]
    for row, line in enumerate(lines):
        stdscr.addstr(row, 0, line[: max(0, curses.COLS - 1)])
    stdscr.refresh()


def key_help(args: argparse.Namespace) -> str:
    if args.key_preset == "robot-plus-x-left-y-up-z":
        return "Left:+y robot-left  Right:-y  Up:+z  Down:-z  f/PgDn:+x robot-forward  b/PgUp:-x"
    if args.key_preset == "robot-minus-x-left-y-up-z":
        return "Legacy inverted-x: Left:+y  Right:-y  Up:+z  Down:-z  f/PgDn:-x  b/PgUp:+x"
    return f"Left/Right: -/+ {args.horizontal_axis}    Up/Down: +/- {args.vertical_axis}"


if __name__ == "__main__":
    raise SystemExit(main())
