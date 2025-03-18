# Team Member: 

from controller import Robot
import numpy as np
import math
import time

class PandaController(Robot):
    def __init__(self):
        super().__init__()
        
        # Setup robot config
        self.TIME_STEP = 32
        self.MAX_VELOCITY = 0.5
        
        # Get all motors and sensors ready
        self.init_robot_components()
        
        # Where the blackboard is - might need to adjust
        self.blackboard_position = [0.5, 0, 0.5]
        
        # Create path points for CU and buffalo
        self.cu_points = self.make_cu_points()
        self.buffalo_points = self.make_buffalo_outline()
        
    def init_robot_components(self):
        # Get all 7 motors for the arm
        self.motors = []
        for i in range(7):
            motor = self.getDevice(f"panda_joint{i+1}")
            motor.setVelocity(self.MAX_VELOCITY)
            self.motors.append(motor)
        
        # Set up the pen (finger joint)
        self.finger = self.getDevice("panda_finger_joint1")
        self.finger.setPosition(0.04)
        
    def make_cu_points(self):
        # Making C shape
        c_points = []
        r = 0.05  # size of letters
        c_center = [self.blackboard_position[0] - 0.08, 
                  self.blackboard_position[1], 
                  self.blackboard_position[2]]
        
        # Create C using arc points
        for t in np.linspace(0.8 * math.pi, -0.8 * math.pi, 20):
            x = c_center[0] + r * math.cos(t)
            z = c_center[2] + r * math.sin(t)
            c_points.append([x, c_center[1], z])
        
        # Making U shape
        u_points = []
        u_center = [self.blackboard_position[0] + 0.08, 
                  self.blackboard_position[1], 
                  self.blackboard_position[2]]
        
        # Left side going down
        for t in np.linspace(0, 1, 10):
            x = u_center[0] - r
            z = u_center[2] - t * r
            u_points.append([x, u_center[1], z])
        
        # Bottom curve
        for t in np.linspace(-math.pi, 0, 10):
            x = u_center[0] + r * math.cos(t)
            z = u_center[2] - r + r * math.sin(t)
            u_points.append([x, u_center[1], z])
        
        # Right side going up
        for t in np.linspace(0, 1, 10):
            x = u_center[0] + r
            z = u_center[2] - r + t * r
            u_points.append([x, u_center[1], z])
        
        # Add a jump point between C and U
        jump_point = [c_center[0] + r + 0.02, c_center[1], c_center[2]]
        
        # Put it all together
        return c_points + [jump_point] + u_points
    
    def make_buffalo_outline(self):
        # Put the buffalo below the CU
        center = [self.blackboard_position[0], 
                self.blackboard_position[1], 
                self.blackboard_position[2] - 0.15]
        
        buffalo_points = []
        
        # Main buffalo body
        body_points = []
        # Modified circle for buffalo body
        for t in np.linspace(0.3, 2*math.pi - 0.3, 30):
            x = center[0] + 0.1 * math.cos(t)
            z = center[2] + 0.07 * math.sin(t)
            
            # Leg area
            if t > math.pi and t < 1.8*math.pi:
                x += 0.03
                z -= 0.01
            # Head area
            if t > 0.3 and t < math.pi/2:
                x += 0.04
            # Tail area
            if t > 5*math.pi/3:
                x -= 0.02
                
            body_points.append([x, center[1], z])
        
        # Adding simplified CU inside
        cu_center = [center[0] - 0.02, center[1], center[2] + 0.01]
        cu_radius = 0.02
        
        # Letter C inside
        c_points = []
        for t in np.linspace(0.7*math.pi, -0.7*math.pi, 10):
            x = cu_center[0] + cu_radius * math.cos(t)
            z = cu_center[2] + cu_radius * math.sin(t)
            c_points.append([x, cu_center[1], z])
        
        # Transition point with pen up
        jump = [cu_center[0] + cu_radius + 0.01, cu_center[1], cu_center[2]]
        
        # Letter U inside
        u_center = [cu_center[0] + 0.04, cu_center[1], cu_center[2]]
        u_points = []
        
        # Left side
        for t in np.linspace(0, 1, 5):
            u_points.append([u_center[0] - cu_radius/2, u_center[1], 
                           u_center[2] - t*cu_radius])
        
        # Bottom curve
        for t in np.linspace(-math.pi, 0, 5):
            x = u_center[0] + (cu_radius/2) * math.cos(t)
            z = u_center[2] - cu_radius + (cu_radius/2) * math.sin(t)
            u_points.append([x, u_center[1], z])
        
        # Right side
        for t in np.linspace(0, 1, 5):
            u_points.append([u_center[0] + cu_radius/2, u_center[1], 
                           u_center[2] - cu_radius + t*cu_radius])
        
        # Put the buffalo together
        buffalo_points = body_points + [jump] + c_points + [jump] + u_points
        
        return buffalo_points
    
    # DH parameters from lab manual
    def fk_solver(self, joints):
        # DH params [a, alpha, d, theta]
        dh = [
            [0, math.pi/2, 0.333, joints[0]],
            [0, -math.pi/2, 0, joints[1]],
            [0, math.pi/2, 0.316, joints[2]],
            [0.0825, math.pi/2, 0, joints[3]],
            [-0.0825, -math.pi/2, 0.384, joints[4]],
            [0, math.pi/2, 0, joints[5]],
            [0.088, 0, 0.107, joints[6]]
        ]
        
        # Calculate transformation matrix
        T = np.eye(4)
        for params in dh:
            a, alpha, d, theta = params
            
            # Standard DH matrix
            T_i = np.array([
                [math.cos(theta), -math.sin(theta)*math.cos(alpha), math.sin(theta)*math.sin(alpha), a*math.cos(theta)],
                [math.sin(theta), math.cos(theta)*math.cos(alpha), -math.cos(theta)*math.sin(alpha), a*math.sin(theta)],
                [0, math.sin(alpha), math.cos(alpha), d],
                [0, 0, 0, 1]
            ])
            
            T = T @ T_i
        
        return T[:3, 3]  # Just need the position part
    
    # Use finite differences to get Jacobian
    def get_jacobian(self, joint_angles):
        h = 1e-6  # Small step for numerical derivative
        J = np.zeros((3, 7))
        
        # Current position
        pos = self.fk_solver(joint_angles)
        
        # Perturb each joint slightly
        for i in range(7):
            # Make a small change to this joint
            test_joints = joint_angles.copy()
            test_joints[i] += h
            
            # New position
            new_pos = self.fk_solver(test_joints)
            
            # Approximate derivative
            J[:, i] = (new_pos - pos) / h
        
        return J
    
    # Calculate joint angles for a target position
    def ik_solver(self, target, current_joints=None):
        # Starting joint config
        if current_joints is None:
            current_joints = np.array([0, 0, 0, -math.pi/2, 0, math.pi/2, 0])
        
        pos = self.fk_solver(current_joints)
        
        # IK solver parameters
        max_iter = 100
        tol = 0.001
        damp = 0.01  # Damping for stability
        
        for iter in range(max_iter):
            # Error between current and target positions
            error = np.array(target) - pos
            error_mag = np.linalg.norm(error)
            
            # Check if we're close enough
            if error_mag < tol:
                return current_joints
            
            # Get Jacobian
            J = self.get_jacobian(current_joints)
            
            # Damped least squares method
            J_T = J.transpose()
            lambda_sq = damp*damp
            inv_term = np.linalg.inv(J @ J_T + lambda_sq * np.eye(3))
            delta_q = J_T @ inv_term @ error
            
            # Take a partial step to prevent overshoot
            step = min(1.0, 0.5 * error_mag)
            current_joints = current_joints + step * delta_q
            
            # Joint limits from Franka manual
            current_joints = np.clip(
                current_joints,
                [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
            )
            
            # Update position after joint change
            pos = self.fk_solver(current_joints)
        
        # If we didn't converge
        print("IK didn't converge! Using best approximation")
        return current_joints
    
    def move_arm(self, target_pos, prev_joints=None):
        # Get the joint angles
        joints = self.ik_solver(target_pos, prev_joints)
        
        # Move each motor
        for i, angle in enumerate(joints):
            self.motors[i].setPosition(angle)
        
        # Let the movement finish
        self.step(self.TIME_STEP * 5)
        
        return joints
    
    def draw(self, points):
        # Move to first point
        prev = self.move_arm(points[0])
        
        # Then draw through all other points
        for p in points[1:]:
            prev = self.move_arm(p, prev)
            
    def run(self):
        print("Starting to draw CU")
        
        # Draw CU letters
        self.draw(self.cu_points)
        
        print("Drawing buffalo logo")
        # Draw buffalo logo
        self.draw(self.buffalo_points)
        
        print("Drawing finished!")

# Start the controller
if __name__ == "__main__":
    controller = PandaController()
    controller.run()
