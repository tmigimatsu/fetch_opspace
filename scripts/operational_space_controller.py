#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np
import PyKDL
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model

import threading
import time

IDX_JOINT_1 = 6
BASE_LINK = "torso_lift_link"
END_LINK = "gripper_link"

def pseudoinverse(A, threshold=0.001):
    U, s, VT = np.linalg.svd(A)
    s_inv = np.zeros(s.shape)
    idx = s > threshold
    s_inv[idx] = 1 / s[idx]
    return np.dot(VT.T, s_inv[:,np.newaxis] * U.T)

class JointSpaceController:

    def __init__(self, freq_control=100, margin_workspace=0.05):
        # Load robot
        urdf_model = URDF.from_parameter_server()
        fetch = kdl_tree_from_urdf_model(urdf_model)
        fetch_arm = fetch.getChain(BASE_LINK, END_LINK)
        self.dof = fetch_arm.getNrOfJoints()

        self.kdl_pos = PyKDL.ChainFkSolverPos_recursive(fetch_arm)
        self.kdl_jac = PyKDL.ChainJntToJacSolver(fetch_arm)
        self.kdl_dyn = PyKDL.ChainDynParam(fetch_arm, PyKDL.Vector(0, 0, -9.81))
        self.kdl_q = PyKDL.JntArray(self.dof)
        self.kdl_A = PyKDL.JntSpaceInertiaMatrix(self.dof)
        self.kdl_J = PyKDL.Jacobian(self.dof)
        self.kdl_x = PyKDL.Frame()

        # self.kdl_G = PyKDL.JntArray(self.dof)
        # self.G = np.zeros((self.dof,))

        # Initialize robot values
        self.lock = threading.Lock()
        self.thread_q = np.zeros((self.dof,))
        self.thread_dq = np.zeros((self.dof,))
        self.thread_tau = np.zeros((self.dof,))

        self.q = np.zeros((self.dof,))
        self.dq = np.zeros((self.dof,))
        self.tau = np.zeros((self.dof,))
        self.x = np.zeros((3,))
        self.quat = np.array([0., 0., 0., 1.])
        self.lim_norm_x = self.compute_workspace() - margin_workspace

        self.A = np.zeros((self.dof, self.dof))
        self.A_inv = np.zeros((self.dof, self.dof))
        self.J = np.zeros((6, self.dof))

        self.q_init = np.array([0., -np.pi/4., 0., np.pi/4, 0., np.pi/2, 0.,]) # Face down
        self.q_tuck = np.array([1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]) # Storage position
        self.q_stow = np.array([1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0])
        self.q_intermediate_stow = np.array([0.7, -0.3, 0.0, -0.3, 0.0, -0.57, 0.0])

        self.x_des = np.array([0.8, 0., 0.35]) # q_init
        self.x_init = np.array([0.8, 0., 0.35]) # q_init
        # self.quat_des = np.array([-0.707, 0., 0.707, 0.]) # Face up
        self.quat_des = np.array([0., 0.707, 0., 0.707]) # Face down

        self.state = "INITIALIZE"
        print("Switching to state: " + self.state)
        self.t_start = time.time()
        self.freq_control = freq_control

        # Initialize pub and sub
        self.pub = rospy.Publisher("arm_controller/joint_torque/command", Float64MultiArray, queue_size=1)
        sub = rospy.Subscriber("joint_states", JointState, lambda joint_states: self.read_joint_sensors(joint_states))

        # Initialize ROS
        rospy.init_node("joint_space_controller")

    def read_joint_sensors(self, joint_states):
        if len(joint_states.position) != 13: # Ignore gripper messages
            return
        with self.lock:
            np.copyto(self.thread_q, joint_states.position[IDX_JOINT_1:IDX_JOINT_1+self.dof])
            np.copyto(self.thread_dq, joint_states.velocity[IDX_JOINT_1:IDX_JOINT_1+self.dof])
            np.copyto(self.thread_tau, joint_states.effort[IDX_JOINT_1:IDX_JOINT_1+self.dof])

    def compute_workspace(self):
        for i in range(self.dof):
            self.kdl_q[i] = 0
        self.kdl_pos.JntToCart(self.kdl_q, self.kdl_x)
        x = np.zeros(3)
        for i in range(3):
            x[i] = self.kdl_x.p[i]
        return np.linalg.norm(x)

    def update_kinematics_dynamics(self):
        # Transfer joint positions to KDL
        for i in range(self.dof):
            self.kdl_q[i] = self.q[i]

        # Compute end-effector position/orientation
        self.kdl_pos.JntToCart(self.kdl_q, self.kdl_x)
        for i in range(3):
            self.x[i] = self.kdl_x.p[i]
        xyzw = self.kdl_x.M.GetQuaternion()
        np.copyto(self.quat, xyzw)

        # Compute Jacobian
        self.kdl_jac.JntToJac(self.kdl_q, self.kdl_J)
        for i in range(6):
            for j in range(self.dof):
                self.J[i,j] = self.kdl_J[i,j]

        # Compute mass matrix
        self.kdl_dyn.JntToMass(self.kdl_q, self.kdl_A)
        for i in range(self.dof):
            for j in range(self.dof):
                self.A[i,j] = self.kdl_A[i,j]
        self.A += np.diag([0., 0., 0., 0.1, 0.2, 0.3, 0.4])
        self.A_inv = np.linalg.inv(self.A)

        # Compute gravity
        # self.kdl_dyn.JntToGravity(self.kdl_q, self.kdl_G)
        # for i in range(self.dof):
        #     self.G[i] = self.kdl_G[i]

    def is_singular(self, J, threshold=0.005):
        return np.linalg.det(J.dot(J.T)) < threshold

    def pd_control(self, x, dx, x_des, kp, kv, dx_max=None):
        x_err = x - x_des
        ddx = -kp * x_err - kv * dx
        if dx_max is not None:
            dx_des = -kp / kv * x_err
            norm_dx_des = np.linalg.norm(dx_des)
            nu = min(1, dx_max / norm_dx_des) if norm_dx_des > 0 else 0
            if nu < 1:
                print(nu)
            dx_err = dx - nu * dx_des
            ddx = -kv * dx_err

        return ddx, x_err

    def position_orientation_control(self, x_des, quat_des, N=None, kp_pos=40, kv_pos=10, kp_ori=10, kv_ori=5, dx_max=0.5):
        if N is None:
            N = np.eye(self.dof)

        # Position and orientation
        J = self.J.dot(N)
        Lambda = pseudoinverse(J.dot(self.A_inv.dot(J.T)))
        J_bar = self.A_inv.dot(J.T).dot(Lambda)
        N = (np.eye(self.dof) - J_bar.dot(J)).dot(N)

        dx_w = J.dot(self.dq)
        E_pinv = np.array([
            [ self.quat[3], -self.quat[2],  self.quat[1], -self.quat[0]],
            [ self.quat[2],  self.quat[3], -self.quat[0], -self.quat[1]],
            [-self.quat[1],  self.quat[0],  self.quat[3], -self.quat[2]]
        ])
        dPhi = -2 * E_pinv.dot(quat_des)
        ddx, x_err = self.pd_control(self.x, dx_w[:3], x_des, kp_pos, kv_pos, dx_max)
        dw = -kp_ori * dPhi - kv_pos * dx_w[3:]
        ddx_dw = np.concatenate((ddx, dw))
        tau = J.T.dot(Lambda.dot(ddx_dw))

        return tau, N, x_err, dPhi

    def joint_space_control(self, q_des, N=None, kp=40, kv=10, dq_max=None):
        ddq, q_err = self.pd_control(self.q, self.dq, q_des, kp, kv, dq_max)

        if N is None:
            tau = self.A.dot(ddq)
        else:
            Lambda = pseudoinverse(N.dot(self.A_inv.dot(N.T)))
            tau = N.T.dot(Lambda.dot(ddq))

        return tau, q_err

    def position_control(self, x_des, N=None, kp=40, kv=10, dx_max=0.5):
        if N is None:
            N = np.eye(self.dof)

        J_v = self.J[:3].dot(N)
        Lambda = pseudoinverse(J_v.dot(self.A_inv.dot(J_v.T)))
        J_v_bar = self.A_inv.dot(J_v.T).dot(Lambda)
        N = (np.eye(self.dof) - J_v_bar.dot(J_v)).dot(N)

        dx = J_v.dot(self.dq)
        ddx, x_err = self.pd_control(self.x, dx, x_des, kp, kv, dx_max)

        tau = J_v.T.dot(Lambda.dot(ddx))

        return tau, N, x_err

    def orientation_control(self, quat_des, N=None, kp=10, kv=5):
        if N is None:
            N = np.eye(self.dof)

        J_w = self.J[3:].dot(N)
        w = J_w.dot(self.dq)
        Lambda = pseudoinverse(J_w.dot(self.A_inv.dot(J_w.T)))
        J_w_bar = self.A_inv.dot(J_w.T.dot(Lambda))
        N = (np.eye(self.dof) - J_w_bar.dot(J_w)).dot(N)

        E_pinv = np.array([
            [ self.quat[3], -self.quat[2],  self.quat[1], -self.quat[0]],
            [ self.quat[2],  self.quat[3], -self.quat[0], -self.quat[1]],
            [-self.quat[1],  self.quat[0],  self.quat[3], -self.quat[2]]
        ])
        dPhi = -2 * E_pinv.dot(quat_des)
        dw = -kp * dPhi - kv * w
        tau = J_w.T.dot(Lambda.dot(dw))

        return tau, N, dPhi

    def run(self):

        ros_command = Float64MultiArray()

        r = rospy.Rate(self.freq_control)
        idx_iter = 0
        t_interval = time.time()
        while not rospy.is_shutdown():

            # Get joint states
            with self.lock:
                np.copyto(self.q, self.thread_q)
                np.copyto(self.dq, self.thread_dq)
                np.copyto(self.tau, self.thread_tau)

            # Compute control torques
            self.update_kinematics_dynamics()
            t_curr = time.time()

            if self.state == "INITIALIZE":
                # Initialize joint space
                tau_des, _, x_err = self.position_control(self.x_init, kp=10)

                # Check for convergence
                if np.linalg.norm(x_err) < 0.1 and np.linalg.norm(self.dq) < 0.1:
                    self.state = "JOINT_SPACE_INIT"
                    print("Switching to state: " + self.state)
                    self.t_start = t_curr

            elif self.state == "JOINT_SPACE_INIT":
                # Initialize joint space
                tau_des, q_err = self.joint_space_control(self.q_init, dq_max=np.pi)

                # Check for convergence
                if np.linalg.norm(q_err) < 0.1:
                    self.state = "OPERATIONAL_SPACE"
                    print("Switching to state: " + self.state)
                    self.t_start = t_curr

            elif self.state == "OPERATIONAL_SPACE":
                # Perform circular trajectory
                x_des = np.copy(self.x_des)
                dt = t_curr - self.t_start
                x_des[0] += 0.3 * np.cos(2*np.pi/10 * dt) - 0.05 + 0.15
                x_des[1] += 0.3 * np.sin(2*np.pi/10 * dt)

                # Compute control torques
                if self.is_singular(self.J):
                    # Drop orientation control
                    if np.linalg.norm(x_des) > self.lim_norm_x:
                        # Restrict x_des to within the workspace radius
                        x_des = x_des / np.linalg.norm(x_des) * self.lim_norm_x
                    tau_x, N, _ = self.position_control(x_des)
                    tau_q, _ = self.joint_space_control(self.q_init, N=N, kp=10, dq_max=None)
                else:
                    # Position/orientation control
                    tau_x, N, _, _ = self.position_orientation_control(x_des, self.quat_des)
                    tau_q, _ = self.joint_space_control(self.q_init, N=N, kp=0, dq_max=None)
                tau_des = tau_x + tau_q

            else:
                # Send nothing
                tau_des = None

            # Publish to ROS
            if tau_des is not None:
                ros_command.data = tau_des.tolist()
                self.pub.publish(ros_command)

            # Track frequency
            # if idx_iter % 100 == 0:
            #     print(t_curr - t_interval)
            #     t_interval = t_curr
            idx_iter += 1
            r.sleep()


if __name__ == '__main__':
    controller = JointSpaceController()
    controller.run()
