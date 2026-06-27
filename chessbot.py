#!/usr/bin/python3

import time
import sys
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from robotiq_gripper import RobotiqGripper
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import pyzed.sl as sl
import math

URI_IP = "192.168.56.101"
AYAL_IP = "192.168.57.101"

A1_URI = [0.08205673493996721, -0.6753187762204925, 0.024844870270488956, 2.247469008346155, -2.146149708890044, 0]
H8_URI = [-0.20489910222067206, -0.4023634743625998, 0.025756493662524194, 2.247613176209329, -2.1463343846561838, 0]
A1_AYAL_ = [-0.4962274477899617, -0.5022420795908198, -0.23857286758367258, -0.03210235266837886, 3.139885740996207, -0.03280161913960377]
H8_AYAL_ = [-0.21527979233210007, -0.22202487265443357, -0.24005978762707658, -0.03210042154743317, 3.1398883430416626, -0.03278482642258741]

############################################### test #################################################################
A1_AYAL = [-0.4962274477899617, -0.5022420795908198, -0.23857286758367258, 0, np.pi, -0.03280161913960377]
H8_AYAL = [-0.21527979233210007, -0.22202487265443357, -0.24005978762707658, 0, np.pi, -0.03278482642258741]
rad = math.atan2(H8_AYAL[1] - A1_AYAL[1], H8_AYAL[0] - A1_AYAL[0])
A1_AYAL[5] = rad-np.pi/4
H8_AYAL[5] = rad-np.pi/4
#print(rad-np.pi/4)
######################################################################################################################
BASE = [0.5155237913131714, -1.0520588916591187, 2.1573708693133753, -2.685061117211813, -1.549094025288717, -1.0686124006854456]

ROBOT_IP = AYAL_IP
BASE_TCP_PORT = 63352
A1_ = A1_AYAL
H8_ = H8_AYAL

grip_size = {
    "queen": 179,  # מלכה
    "pawn": 196,   # רגלי
    "king": 176,   # מלך
    "rook": 182,   # צריח
    "knight": 204, # פרש
    "bishop": 190, # רץ
}

X, Y, Z, RX, RY, RZ = 0,1,2,3,4,5
GRIP_RELEASE_OFFSET = 30
GRIP_RELEASE_HEIGHT = 0.005
CLOSED = 255
OPENED = 0
HALF_OPENED = 140
OFFSET_TO_TABLE_HEIGHT = -0.02





clicked_point = None
point_cloud = sl.Mat()


R = np.array([[-0.99954817, -0.02010487, -0.02234386],
              [ 0.00272617,  0.67966745, -0.73351532],
              [ 0.02993362, -0.73324481, -0.67930555]])

t = np.array([-0.44277486, 0.1592128, 0.1834975 ])




def reset_gripper(robot_ip=ROBOT_IP, base_tcp_port=BASE_TCP_PORT):
    print("Reset arg detected: resetting and activating gripper...")
    g = None
    try:
        g = RobotiqGripper()
        g.connect(robot_ip, base_tcp_port)
        try:
            g._reset()
        except Exception as exc:
            print("Warning during gripper reset:", exc)
        try:
            g.activate()
        except Exception as exc:
            print("Warning during gripper activate:", exc)
        print("Gripper reset+activate completed.")
    except Exception as exc:
        print("Error communicating with gripper:", exc)
    finally:
        if g is not None:
            try:
                g.disconnect()
            except Exception:
                pass
    sys.exit(0)
def move_to_start_postion():
    with ChessBot(robot_ip=ROBOT_IP, base_tcp_port=BASE_TCP_PORT, A1=A1_, H8=H8_) as robot:
        robot.move_to(robot.start_position, z=robot.safe_height)
    sys.exit(0)
def grip_close():
    with ChessBot(robot_ip=ROBOT_IP, base_tcp_port=BASE_TCP_PORT, A1=A1_, H8=H8_) as robot:
        robot.set_gripper(CLOSED)
    sys.exit(0)
def grip_open():
    with ChessBot(robot_ip=ROBOT_IP, base_tcp_port=BASE_TCP_PORT, A1=A1_, H8=H8_) as robot:
        robot.set_gripper(OPENED)
    sys.exit(0)
def print_position():
    with ChessBot(robot_ip=ROBOT_IP, base_tcp_port=BASE_TCP_PORT, A1=A1_, H8=H8_) as robot:
        print(robot.pose)
    sys.exit(0)
def align_position():
    with ChessBot(robot_ip=ROBOT_IP, base_tcp_port=BASE_TCP_PORT, A1=A1_, H8=H8_) as robot:
        robot.rtde_c.moveJ(BASE, 1, 0.5)
    sys.exit(0)
def get_grip():
    with ChessBot(robot_ip=ROBOT_IP, base_tcp_port=BASE_TCP_PORT, A1=A1_, H8=H8_) as robot:
        print(robot.get_gripper())
    sys.exit(0)
def print_joints():
    with ChessBot(robot_ip=ROBOT_IP, base_tcp_port=BASE_TCP_PORT, A1=A1_, H8=H8_) as robot:
        print(list(robot.rtde_r.getActualQ()))
    sys.exit(0)






class ChessBot:
    def __init__(self, robot_ip=None, base_tcp_port=None, speed=0.1, acceleration=0.1, A1 = None, H8 = None):
        self.robot_ip = robot_ip
        self.base_tcp_port = base_tcp_port
        self.speed = speed
        self.acceleration = acceleration
        self.rtde_c = None
        self.rtde_r = None
        self.gripper = None
        self.A1 = A1
        self.H8 = H8
        self.step_right = [0, 0]
        self.step_up = [0, 0]
        self.floor_height = 0
        self.sky_height = 0
        self.safe_height = 0.15
        self.start_position = [0, 0, 0, 0, 0, 0]
        self.positions = {}
        self.grip_height = {}
        self.release_lying_height = {}
        self.down_orientation = [2.0107452890463056, -2.3743186369030598, -0.0762631964824858]
        self.free_platform = [0,0,0,0,0,0]
        self.table_height = 0
        self.cube_pose = [0,0,0,0,0,0]

    def __enter__(self):
        self.rtde_c = RTDEControlInterface(self.robot_ip)
        self.rtde_r = RTDEReceiveInterface(self.robot_ip)
        self.gripper = RobotiqGripper()
        self.gripper.connect(self.robot_ip, self.base_tcp_port)
        time.sleep(0.1)
        self.calibrate_board_positions(self.A1, self.H8)
        #self.rtde_c.moveL(self.start_position, self.speed, self.acceleration)
        if not self.gripper.is_active():
            self.gripper.activate()
        return self

    # return a position moved right and up by the given number of centimeters, can also use negative numbers to move left and down
    # must before use the calibrate_board_positions function at least once to set the step_right and step_up values
    def move_on_chessboard(self, current_pos, right=0, up=0):
        position = current_pos.copy()
        position[X] = position[X] + (self.step_right[X] / 4.0) * right + (self.step_up[X] / 4.0) * up
        position[Y] = position[Y] + (self.step_right[Y] / 4.0) * right + (self.step_up[Y] / 4.0) * up
        return position

    def calibrate_board_positions(self, a1=None, h8=None):
       
        if a1 is None:
            a1 = A1
        if h8 is None:
            h8 = H8

        dx = h8[X] - a1[X]
        dy = h8[Y] - a1[Y]
        h1 = [a1[X] + (dx + dy) / 2.0, a1[Y] + (dy - dx) / 2.0]

        self.step_right[0] = (h1[X] - a1[X]) / 7.0
        self.step_right[1] = (h1[Y] - a1[Y]) / 7.0
        self.step_up[0] = -self.step_right[Y]
        self.step_up[1] = self.step_right[X]

        self.down_orientation = [(h8[RX] + a1[RX]) / 2, (h8[RY] + a1[RY]) / 2, (h8[RZ] + a1[RZ]) / 2]
        self.floor_height = (h8[Z] + a1[Z]) / 2
        self.sky_height = self.floor_height + 0.3

        tmp_pos = [a1[X], a1[Y]]
        for row in range(1, 9):
            for col in "abcdefgh":
                square = f"{col}{row}"
                self.positions[square] = tmp_pos + [self.floor_height] + self.down_orientation
                tmp_pos = self.move_on_chessboard(tmp_pos, right=4, up=0)
            tmp_pos = self.move_on_chessboard(tmp_pos, right=-32, up=4)

        self.start_position = self.move_on_chessboard(self.positions['d5'], right = 2, up = -2)
        self.start_position[Z] = self.sky_height

        self.grip_height["queen"] = self.floor_height + 0.04
        self.grip_height["pawn"] = self.floor_height + 0.025 
        self.grip_height["king"] = self.floor_height + 0.035
        self.grip_height["rook"] = self.floor_height + 0.025
        self.grip_height["knight"] = self.floor_height + 0.03
        self.grip_height["bishop"] = self.floor_height + 0.03
        
        self.release_lying_height["queen"] = self.floor_height + 0.065
        self.release_lying_height["pawn"] = self.floor_height + 0.03
        self.release_lying_height["king"] = self.floor_height + 0.06
        self.release_lying_height["rook"] = self.floor_height + 0.040
        self.release_lying_height["knight"] = self.floor_height + 0.045
        self.release_lying_height["bishop"] = self.floor_height + 0.028

        self.safe_height = 0.15 + self.floor_height
        self.free_platform = self.move_on_chessboard(self.positions['a8'], right=-8.5, up=2)[0:2] + [self.floor_height + 0.042] + self.down_orientation
        self.table_height = self.floor_height - 0.02

        
        self.cube_pose = self.move_on_chessboard(self.positions['a8'], up=9, right=-0.5)
        self.cube_pose[Z] += 0.038
        self.cube_pose[3:] = self.get_rotated_tcp_orientation(base_orientation = self.cube_pose, Rz = 90)
        
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self.rtde_c is not None:
                self.rtde_c.stopScript()
        except Exception:
            pass
        try:
            if self.gripper is not None:
                self.gripper.disconnect()
        except Exception:
            pass
        try:
            if self.rtde_c is not None:
                self.rtde_c.disconnect()
        except Exception:
            pass
        try:
            if self.rtde_r is not None:
                self.rtde_r.disconnect()
        except Exception:
            pass


    def normalize_pos(self, pos):
        if isinstance(pos, str):
            return self.positions[pos]
        if isinstance(pos, (np.ndarray, tuple)):
            pos = list(map(float, pos))
        if not isinstance(pos, list):
            raise Exception("expected a list")
        if len(pos) == 3:
            return pos + self.down_orientation
        elif len(pos) == 6:
            return pos
        else:
            raise Exception("bad length")

    @property
    def pose(self):
        return self.rtde_r.getActualTCPPose()

    # return a rotated version vector of curent position by tcp
    def get_rotated_tcp_orientation(self, base_orientation = None, Rx=0, Ry=0, Rz=0):
        if base_orientation is None:
            base_orientation = self.pose
        base_orientation = base_orientation[-3:] 
        rot_base = Rotation.from_rotvec(base_orientation)
        rot_local = Rotation.from_euler('xyz', [Rx, Ry, Rz], degrees=True)
        new_rot = rot_base * rot_local
        return  new_rot.as_rotvec().tolist()
    
    def calculate_target_pose(self, pose=None, x=None, y=None, z=None, rx=None, ry=None, rz=None, 
                              orientation = None, dx=None, dy=None, dz=None, drx=None, dry=None, drz=None):
        if pose is None:
            pose = self.pose

        # Handle orientation case
        if orientation is not None:
            # Check if any x, y, z are provided
            other_params = any(v is not None for v in (rx, ry, rz, drx, dry, drz))
            if other_params:
                raise ValueError("Cannot mix orientation with x, y, z parameters")
        
        # Use unified modify_pose for all position/rotation parameters
        target_pose = self.modify_pose(pose, x=x, y=y, z=z, rx=rx, ry=ry, rz=rz,
                                        dx=dx, dy=dy, dz=dz, drx=drx, dry=dry, drz=drz)
        
        if orientation is not None:
            target_pose[3:] = orientation

        return target_pose

    def move_to(self, pose=None, x=None, y=None, z=None, rx=None, ry=None, rz=None, orientation = None,
                dx=None, dy=None, dz=None, drx=None, dry=None, drz=None, speed = None, acceleration = None):
        """Move to a target pose. Accepts absolute, relative, or mixed arguments.
        
        Absolute parameters (x, y, z, rx, ry, rz) set the coordinate directly.
        Relative parameters (dx, dy, dz, drx, dry, drz) add to the current/given pose.
        Can also provide orientation directly or mix absolute and relative parameters.
        """
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration
        target_pose = self.calculate_target_pose(pose=pose, x=x, y=y, z=z, rx=rx, ry=ry, rz=rz, orientation = orientation,
                                                dx=dx, dy=dy, dz=dz, drx=drx, dry=dry, drz=drz)
        self.rtde_c.moveL(target_pose, speed, acceleration)
        return target_pose

    def set_gripper(self, position=None, close_by=None, open_by=None, speed=120, force=0, wait = True):
        if position is None and close_by is None and open_by is None:
            raise Exception("expected position or close_by or open_by")
        if position is None:
            position = self.get_gripper()
        if close_by is not None:
            position += close_by
        if open_by is not None:
            position -= open_by
        if wait == True:
            self.gripper.move_and_wait_for_pos(position, speed, force)
        else:
            self.gripper.move(position, speed, force)

    def get_gripper(self):
        return self.gripper.get_current_position()
    @staticmethod
    def modify_pose(pose, x=None, y=None, z=None, rx=None, ry=None, rz=None,
                    dx=None, dy=None, dz=None, drx=None, dry=None, drz=None):
        """Modify a pose with absolute and/or relative parameters.
        
        Absolute parameters (x, y, z, rx, ry, rz) set the coordinate directly.
        Relative parameters (dx, dy, dz, drx, dry, drz) add to the coordinate.
        If both absolute and relative are provided for the same coordinate, they are summed.
        """
        modified = pose.copy()
        
        # Process X coordinate (absolute and/or relative)
        if x is not None or dx is not None:
            value = pose[X]
            if x is not None:
                value = x
            if dx is not None:
                value += dx
            modified[X] = value
        
        # Process Y coordinate (absolute and/or relative)
        if y is not None or dy is not None:
            value = pose[Y]
            if y is not None:
                value = y
            if dy is not None:
                value += dy
            modified[Y] = value
        
        # Process Z coordinate (absolute and/or relative)
        if z is not None or dz is not None:
            value = pose[Z]
            if z is not None:
                value = z
            if dz is not None:
                value += dz
            modified[Z] = value
        
        # Process RX rotation (absolute and/or relative)
        if rx is not None or drx is not None:
            value = pose[RX]
            if rx is not None:
                value = rx
            if drx is not None:
                value += drx
            modified[RX] = value
        
        # Process RY rotation (absolute and/or relative)
        if ry is not None or dry is not None:
            value = pose[RY]
            if ry is not None:
                value = ry
            if dry is not None:
                value += dry
            modified[RY] = value
        
        # Process RZ rotation (absolute and/or relative)
        if rz is not None or drz is not None:
            value = pose[RZ]
            if rz is not None:
                value = rz
            if drz is not None:
                value += drz
            modified[RZ] = value
        
        return modified

    def mov_chess_piece(self, type=None, start_pos=None, end_pos=None, speed=None, acceleration=None, rz_rotation_start=None, rz_rotation_end=None, move_to_start=True):
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration
        if self.pose[Z] < self.safe_height:
            self.move_to(z=self.safe_height)

        # update chess board locations
        start_pos = self.normalize_pos(start_pos)
        end_pos = self.normalize_pos(end_pos)
        if rz_rotation_start is not None:
            start_pos[3:6] = self.get_rotated_tcp_orientation(start_pos,Rz=rz_rotation_start)
        if rz_rotation_end is not None:
            end_pos[3:6] = self.get_rotated_tcp_orientation(end_pos,Rz=rz_rotation_end)

        self.set_gripper(grip_size[type] - GRIP_RELEASE_OFFSET,wait=False) #  open the gripper

        # move to first spot
        self.move_to(start_pos, z=self.safe_height, speed=speed, acceleration=acceleration) 
        self.move_to(z=self.grip_height[type])

        self.set_gripper(CLOSED) # grip the piece

        # move to end spot
        self.move_to(start_pos, z=self.safe_height)
        self.move_to(end_pos, z=self.safe_height, speed=speed, acceleration=acceleration)
        self.move_to(z=self.grip_height[type] + GRIP_RELEASE_HEIGHT)
        
        self.set_gripper(self.get_gripper() - GRIP_RELEASE_OFFSET) # release the piece
        
        # return to start postion
        self.move_to(z=self.safe_height)
        if move_to_start:
            self.move_to(self.start_position, z=self.safe_height, speed=speed, acceleration=acceleration)

    def move_smooth_path___experimental(self, steps, blend_radius=0.03, speed=None, acceleration=None):
        
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration

        path = []
        current_pose = self.pose 

        for i, step in enumerate(steps):
            step_kwargs = step.copy()
            if 'pose' not in step_kwargs: # save current pose as base pose
                step_kwargs['pose'] = current_pose
                
            next_pose = self.calculate_target_pose(**step_kwargs)
            radius = 0.0 if i == len(steps) - 1 else blend_radius
            path.append(next_pose + [speed, acceleration, radius]) # build the path
            
            current_pose = next_pose
            
        self.rtde_c.moveL(path)
        return current_pose

    def mov_chess_piece___experimental(self, type=None, start_pos=None, end_pos=None, blend_radius = 0.05):
        if self.pose[Z] < self.safe_height:
            self.move_to(z=self.safe_height)

        self.set_gripper(grip_size[type] - GRIP_RELEASE_OFFSET, wait=False) #  release the piece

        path_steps1 = [
            {'pose': [start_pos[X], start_pos[Y], self.safe_height] + self.get_rotated_tcp_orientation(start_pos,Rz=45)},
            {'z': self.grip_height[type]}, 
        ]
        self.move_smooth_path(path_steps1, blend_radius=blend_radius) # move smoothly
        
        self.set_gripper(grip_size[type]) # grip the piece

        path_steps2 = [
            {'pose': [start_pos[X], start_pos[Y], self.safe_height] + self.get_rotated_tcp_orientation(start_pos,Rz=45), 'z':self.safe_height},
            {'pose': [end_pos[X], end_pos[Y], self.safe_height] + self.get_rotated_tcp_orientation(end_pos,Rz=45), 'z':self.safe_height}, 
            {'z': self.grip_height[type] + 0.005}
        ]
        self.move_smooth_path(path_steps2, blend_radius=blend_radius, speed=0.05) # move smoothly
        self.move_to(z=self.grip_height[type] + GRIP_RELEASE_HEIGHT)
        
        self.set_gripper(grip_size[type] - GRIP_RELEASE_OFFSET) # release the piece
        
        path_steps3 = [
            {'z':self.safe_height},
            {'pose': self.start_position}, 
        ]
        self.move_smooth_path(path_steps3, blend_radius=blend_radius, speed=0.05) # move smoothly

    def camera_vector_to_robot_vector(self, camera_vector):
        #R, t = estimate_transform(camera_points, robot_points)
        return list(map(float, R @ camera_vector + t)) + self.down_orientation

    @staticmethod
    def weighted_avg(x, y, x_bias):
        if isinstance(x, list):
            return [weighted_avg(a, b, x_bias) for a, b in zip(x, y)]
        return x*x_bias + y*(1-x_bias)

    def pick_up_dead_piece(self, type, state, end_pos):
        ############################################################### add knight support
        # get robot postions from the interactable camera
        base_point, head_point = get_base_and_head_camera_points()
        base_robot =  R @ base_point + t
        head_robot = R @ head_point + t

        # fix alignment
        #base_robot[Y]-=0.01
        #head_robot[Y]-=0.01
        #base_robot[X]+=0.01
        #head_robot[X]+=0.01

        # calculate middle position
        dx = head_robot[0] - base_robot[0]
        dy = head_robot[1] - base_robot[1]
        dz = math.degrees(math.atan2(dx, dy))

        if state == 'standing':
            bias = 0.5
        elif type == 'bishop':
            bias = 0.5
        elif type == 'knight':
            bias = 0.7
        else:
            bias = 0.6
        middle_position = [
            *ChessBot.weighted_avg(head_robot[:2], base_robot[:2], bias),
            0,
            #*self.get_rotated_tcp_orientation(Rz=dz+90)
            *self.get_rotated_tcp_orientation(Rz=dz+180)
        ]
        
        # fix height
        if middle_position[Z] < self.safe_height:
            self.move_to(z=self.safe_height)

        self.set_gripper(HALF_OPENED,wait=False)

        self.move_to(middle_position, z=self.safe_height)
        if state == 'lying':
            self.move_to(z=self.table_height+0.0015)

            self.set_gripper(CLOSED) # grip the piece

            self.move_to(z=self.safe_height)
            #self.move_to(self.cube_pose, z=self.safe_height)
            self.move_to(self.cube_pose, z=self.safe_height, 
                        orientation = self.get_rotated_tcp_orientation(base_orientation = self.cube_pose, Rz=-180))

            # rotate slowly
            self.move_to(orientation = self.get_rotated_tcp_orientation(Rx=85))
            self.move_to(z=self.cube_pose[Z] - self.floor_height + self.grip_height[type] + 0.00)
            
            # release standing piece
            self.set_gripper(open_by=GRIP_RELEASE_OFFSET)

            # straighten the arm back
            
            self.move_to(self.cube_pose,dx=-0.007, z=self.cube_pose[Z] - self.floor_height + self.grip_height[type]- 0.01,
                orientation = self.get_rotated_tcp_orientation(base_orientation = self.cube_pose, Rx=0,Rz=-180))

            self.set_gripper(CLOSED) # grip the piece

            self.move_to(dz=0.05)#, orientation = self.get_rotated_tcp_orientation(Rz=180)) # raise the arm

        elif state == 'standing':
            ##################################################### add knight support
            self.move_to([middle_position[X], middle_position[Y], self.grip_height[type]+OFFSET_TO_TABLE_HEIGHT] + self.down_orientation)
            
            self.set_gripper(CLOSED)
            
            self.move_to(z=self.safe_height)
            
        else:
            print("error on state")
            return 
            
        
        # put back on the chess board
        self.move_to(
            self.positions[end_pos],
            z=self.safe_height,
            orientation =
                self.get_rotated_tcp_orientation(Rz=90)
                if state == 'lying' and type == 'knight' and self.get_gripper() < grip_size['knight']-7
                else None
        )
        self.move_to(z=self.grip_height[type] + GRIP_RELEASE_HEIGHT)

        self.set_gripper(open_by=GRIP_RELEASE_OFFSET)

        self.move_to(z=self.safe_height)
        self.rtde_c.moveJ(BASE, 1, 0.5)
        self.move_to(self.start_position, z=self.safe_height)
        
    def capture_piece(self, type, start_pos, end_pos = None, rz_start=None, move_to_start=True):
        if end_pos is None:
            end_pos = list(map(float, get_head_camera_point())) + self.down_orientation
        start_pos = self.normalize_pos(start_pos)
        end_pos = self.normalize_pos(end_pos)

        self.move_to(start_pos, z=self.safe_height, orientation=self.down_orientation)

        self.set_gripper(grip_size[type] - GRIP_RELEASE_OFFSET,wait=False) #  open the gripper

        self.move_to(z=self.grip_height[type])

        self.set_gripper(CLOSED)

        self.move_to(z=self.safe_height)
        self.move_to(end_pos, z=self.safe_height)
        self.move_to(z=self.grip_height[type]+OFFSET_TO_TABLE_HEIGHT)

        self.set_gripper(open_by=GRIP_RELEASE_OFFSET)

        # return to start position
        self.move_to(z=self.safe_height)
        if move_to_start:
            self.move_to(self.start_position, z=self.safe_height)

    def move_and_capture_piece(self, capturer, captured, empty_pos=None):
        (capturer_type, capturer_pos) = capturer
        (captured_type, captured_pos) = captured
        capturer_pos = self.normalize_pos(capturer_pos)
        captured_pos = self.normalize_pos(captured_pos)

        self.capture_piece(captured_type, captured_pos, empty_pos, move_to_start=False)
        self.mov_chess_piece(capturer_type, capturer_pos, captured_pos)

######################## zed camera #########################
def estimate_transform(camera_points, robot_points):
    """
    camera_points: Nx3 numpy array
    robot_points:  Nx3 numpy array

    Returns:
        R (3x3 rotation matrix)
        t (3-vector translation)
    """

    assert camera_points.shape == robot_points.shape
    assert camera_points.shape[1] == 3

    # Centroids
    centroid_cam = np.mean(camera_points, axis=0)
    centroid_robot = np.mean(robot_points, axis=0)

    # Center points
    cam_centered = camera_points - centroid_cam
    robot_centered = robot_points - centroid_robot

    # Covariance matrix
    H = cam_centered.T @ robot_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # Reflection correction
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_robot - R @ centroid_cam

    return R, t

def mouse_callback(event, x, y, flags, param):
    global clicked_point

    if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)

def get_base_and_head_camera_points():
    global clicked_point, point_cloud
    base_point = None
    head_point = None

    # Create camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED")
        return

    image = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    cv2.namedWindow("ZED")
    cv2.setMouseCallback("ZED", mouse_callback)

    while head_point==None:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Point cloud
            zed.retrieve_measure(
                point_cloud,
                sl.MEASURE.XYZ
            )
            frame = image.get_data()
            if clicked_point is not None:
                x, y = clicked_point
                err, point3d = point_cloud.get_value(x, y)
                if err == sl.ERROR_CODE.SUCCESS:
                    X_, Y_, Z_ = point3d[:3]
                    if np.isfinite(X_) and np.isfinite(Y_) and np.isfinite(Z_):
                        print(f"Pixel ({x}, {y})")
                        print(f"3D point: X={X_:.3f}, Y={Y_:.3f}, Z={Z_:.3f} meters")
                        if(base_point is None):
                            base_point = [X_, Y_, Z_]
                            print(f"base_point is set")
                        else:
                            head_point = [X_, Y_, Z_]
                            print(f"head_point is set")
                    else:
                        print("Invalid depth at this pixel")

                clicked_point = None
            cv2.imshow("ZED", frame)
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

    



    zed.close()

    cv2.destroyAllWindows()

    return base_point, head_point

def get_head_camera_point():
    global clicked_point, point_cloud
    base_point = None
    head_point = None

    # Create camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED")
        return

    image = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    cv2.namedWindow("ZED")
    cv2.setMouseCallback("ZED", mouse_callback)

    while head_point==None:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Point cloud
            zed.retrieve_measure(
                point_cloud,
                sl.MEASURE.XYZ
            )
            frame = image.get_data()
            if clicked_point is not None:
                x, y = clicked_point
                err, point3d = point_cloud.get_value(x, y)
                if err == sl.ERROR_CODE.SUCCESS:
                    X_, Y_, Z_ = point3d[:3]
                    if np.isfinite(X_) and np.isfinite(Y_) and np.isfinite(Z_):
                        print(f"Pixel ({x}, {y})")
                        print(f"3D point: X={X_:.3f}, Y={Y_:.3f}, Z={Z_:.3f} meters")
                        if(False):
                            base_point = [X_, Y_, Z_]
                            print(f"base_point is set")
                        else:
                            head_point = [X_, Y_, Z_]
                            print(f"head_point is set")
                    else:
                        print("Invalid depth at this pixel")

                clicked_point = None
            cv2.imshow("ZED", frame)
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

    zed.close()

    cv2.destroyAllWindows()

    
    #R, t = estimate_transform(camera_points, robot_points)
    head_robot = R @ head_point + t

    # fix alignment
    #head_robot[Y]-=0.01
    #head_robot[X]+=0.01

    return head_robot




def main():
    with ChessBot(robot_ip=ROBOT_IP, base_tcp_port=BASE_TCP_PORT, A1=A1_, H8=H8_, speed=0.5) as robot:
       # מלך, מלכה, רץ, פרש, צריח, רגלי = king, queen, bishop, knight, rook, pawn
        
        print("starting session")
        robot.move_to(robot.start_position, z=robot.safe_height)
        
        robot.move_to(robot.positions['a1'], dz = 0.005)
        #robot.pick_up_dead_piece('queen', 'lying', 'g5')




        
        
        

        #robot.move_to(robot.start_position, z=robot.safe_height)
        print("end of session")

        
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "reset":
            reset_gripper()
        if sys.argv[1].lower() == "start":
            move_to_start_postion()
        if sys.argv[1].lower() == "print":
            print_position()
        if sys.argv[1].lower() == "close":
            grip_close()
        if sys.argv[1].lower() == "open":
            grip_open()
        if sys.argv[1].lower() == "align":
            align_position()
        if sys.argv[1].lower() == "grip":
            get_grip()
        if sys.argv[1].lower() == "joints":
            print_joints()
    else:
        main()



#print(robot.positions['a1'][0:3])
        #print(robot.positions['c2'][0:3])
        #print(robot.positions['e3'][0:3])
        #print(robot.positions['f6'][0:3])
        #print(robot.positions['h7'][0:3])
        #print(robot.positions['h3'][0:3])
        #print(robot.positions['b7'][0:3])
        #print(robot.positions['c6'][0:3])

        # with open("output", "a") as f:
        #     for i in range(0, 8 + 11 + 1, 2):
        #         base, head = get_base_and_head_camera_points()
        #         print(f"point {i}:", list(map(float, base)))
        #         print(f"point {i + 1}:", list(map(float, head)))
        #         print(f"point {i}:", list(map(float, base)), file=f)
        #         print(f"point {i + 1}:", list(map(float, head)), file=f)
        #         f.flush()
        '''
        
        # Press any key 12 times to print robot arm position
        print("\nPress Enter 12 times to print robot arm position. Press Ctrl+C to exit.")
        key_count = 0
        try:
            while True:
                input()  # Wait for user to press Enter
                current_pose = robot.pose
                print(f"\n[Keypress {key_count}] Current Robot Arm Position:")
                print(f"  Position (XYZ): {current_pose[:3]}")
                key_count += 1
                
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
        '''
    #robot.pick_up_dead_piece("queen", "lying", "a1")
    #robot.move_and_capture_piece(("pawn", "b2"), ("queen","c3"))