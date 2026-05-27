import numpy as np
import math
import rby1_sdk as rby
import logging
import argparse
import sys

def make_transform(data):
    # data: [x, y, z, roll, pitch, yaw] (x,y,z in meters, r,p,y in degrees)
    x, y, z = data[0], data[1], data[2] 
    roll = data[3] * math.pi / 180
    pitch = data[4] * math.pi / 180
    yaw = data[5] * math.pi / 180
    
    cr = math.cos(roll); sr = math.sin(roll)
    cp = math.cos(pitch); sp = math.sin(pitch)
    cy = math.cos(yaw); sy = math.sin(yaw)
    
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = cy * cp
    m[0, 1] = sr * sp * cy - cr * sy
    m[0, 2] = cr * sp * cy + sr * sy
    m[0, 3] = x
    
    m[1, 0] = sy * cp
    m[1, 1] = sr * sp * sy + cr * cy
    m[1, 2] = cr * sp * sy - sr * cy
    m[1, 3] = y
    
    m[2, 0] = -sp
    m[2, 1] = cp * sr
    m[2, 2] = cp * cr
    m[2, 3] = z
    
    return m

def compute_fk(robot, ee_link, base_link="link_torso_5"):
    robot_state = robot.get_state()
    q_full = robot_state.position
    dyn_model = robot.get_dynamics()
    dyn_state = dyn_model.make_state(
        [base_link, ee_link],
        robot.model().robot_joint_names
    )
    dyn_state.set_q(q_full)
    dyn_model.compute_forward_kinematics(dyn_state)
    return dyn_model.compute_transformation(dyn_state, 0, 1)

def calc_t5_to_object(robot, cam_to_object_tf):
    t5_to_head_tf = compute_fk(robot, "link_head_1", "link_torso_5")
    head_to_cam_tf = make_transform([0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
    t5_to_object_tf = t5_to_head_tf @ head_to_cam_tf @ cam_to_object_tf
    return t5_to_object_tf



# 이 아래는 확인용 로봇실행코드
def initialize_robot(address, model, power=".*", servo=".*"):
    robot = rby.create_robot(address, model)
    if not robot.connect():
        logging.error(f"Failed to connect robot {address}")
        sys.exit(1)
    if not robot.is_power_on(power):
        if not robot.power_on(power):
            logging.error(f"Failed to turn power ({power}) on")
            sys.exit(1)
    if not robot.is_servo_on(servo):
        if not robot.servo_on(servo):
            logging.error(f"Failed to servo ({servo}) on")
            sys.exit(1)
    if robot.get_control_manager_state().state in [
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ]:
        if not robot.reset_fault_control_manager():
            logging.error(f"Failed to reset control manager")
            sys.exit(1)
    if not robot.enable_control_manager():
        logging.error(f"Failed to enable control manager")
        sys.exit(1)
    return robot

def main(address, model_name, power, servo):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s"
    )
    
    logging.info(f"Initializing robot {address}...")
    robot = initialize_robot(address, model_name, power, servo)
    logging.info("Robot initialized successfully.")
    
    try:
        # Get relative coordinate system of head from T5
        t5_to_head_tf = compute_fk(robot, "link_head_1", "link_torso_5")
        
        print("\n=== Current Pose (t5_to_head_tf) ===")
        print(t5_to_head_tf)
        
        # Dummy camera-to-object transform (Identity matrix) for testing
        cam_to_object_tf = np.eye(4, dtype=np.float64)
        cam_to_object_tf[0:3, 3] = np.array([0.0, 0.0, 0.0])
        t5_to_object_tf = calc_t5_to_object(robot, cam_to_object_tf)
        
        print("\n=== Calculated t5_to_object_tf (assuming identity camera-to-object transform) ===")
        print(t5_to_object_tf)
        
    finally:
        logging.info("Disconnecting from the robot...")
        robot.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="24_demo_motion")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument("--model", type=str, default='a', help="Robot Model Name (default: 'a')")
    parser.add_argument(
        "--power",
        type=str,
        default=".*",
        help="Power device name regex pattern (default: '.*')",
    )
    parser.add_argument(
        "--servo",
        type=str,
        default=".*",
        help="Servo name regex pattern (default: '.*')",
    )
    args = parser.parse_args()

    main(address=args.address, model_name=args.model, power=args.power, servo=args.servo)