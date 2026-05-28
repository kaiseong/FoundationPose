import rby1_sdk as rby
import numpy as np


def main():
    robot = rby.create_robot("localhost:50051", "m")
    if not robot.connect():
        print("Failed to connect to the robot")
        exit(1)
    if not robot.is_power_on(".*"):
        if not robot.power_on(".*"):
            print("Failed to power on")
            exit(1)
    if not robot.is_servo_on(".*"):
        if not robot.servo_on(".*"):
            print("Failed to servo on")
            exit(1)
    if robot.get_control_manager_state().state in [
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ]:
        if not robot.reset_fault_control_manager():
            print("Failed Control Manager Reset")
            exit(1)
    if not robot.enable_control_manager():
        print("Failed enable control manager")
        exit(1)

    dyn_model = robot.get_dynamics()
    robot_state = robot.get_state()
    dyn_state = dyn_model.make_state(["link_torso_5", "ee_right"], robot.model().robot_joint_names)
    dyn_state.set_q(robot_state.position)
    dyn_model.compute_forward_kinematics(dyn_state)

    target_t5_T_ee = np.asarray(dyn_model.compute_transformation(dyn_state, 0, 1), dtype=np.float64)
    target_t5_T_ee[:3, 3] += np.array([0.02, 0.0, 0.0], dtype=np.float64)


    rc = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder().set_right_arm_command(
                rby.CartesianCommandBuilder()
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1.0))
                .set_minimum_time(1.0)
                .set_stop_joint_position_tracking_error(0)
                .set_stop_position_tracking_error(0)
                .set_stop_orientation_tracking_error(0)
                .add_target("link_torso_5", "ee_right", target_t5_T_ee, 1.0, np.pi / 2.0, 1.0)
            )
        )
    )
    feedback = robot.send_command(rc, 10).get()
    print(feedback)
    print(robot.get_state)


if __name__ == "__main__":
    main()
