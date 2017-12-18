import numpy as np
import math

class RobotArm2D:
    '''RobotArm2D([xRoot=0, yRoot=0])

        INPUT ARGUMENTS:

        xRoot, yRoot (optional): x and y coordinates of the root joint.
            Both default to 0 if not set.

        INSTANCE VARIABLES:

        thetas: 1D array of joint angles; contains N elements, one per joint.
        joints: 4 x N array of joint coordinates; each column is a vector
            (column 0 is the root joint and column N-1 is the end effector).
        lengths: list of arm link lengths, containing N elements, where
            lengths[0] is the first link and lengths[N-1] is the last link,
            terminating at the end effector.
    '''
    def __init__(self, **kwargs):
        self.xRoot = kwargs.get('xRoot', 0)
        self.yRoot = kwargs.get('yRoot', 0)
        self.thetas = np.array([[]], dtype=np.float)
        self.joints = np.array([[self.xRoot, self.yRoot, 0, 1]], dtype=np.float).T
        self.lengths = []

    def add_revolute_link(self, **kwargs):
        '''add_revolute_link(length[, thetaInit=0])
            Add a revolute joint to the arm with a link whose length is given
            by required argument "length". Optionally, the initial angle
            of the joint can be specified.
        '''
        self.joints = np.append(self.joints, np.array([[0,0,0,1]]).T, axis=1)
        self.lengths.append(kwargs['length'])
        self.thetas = np.append(self.thetas, kwargs.get('thetaInit', 0))

    def get_transformation_matrix(self, theta, x, y):
        '''get_transformation_matrix(theta, x, y)
            Returns a 4x4 transformation matrix for a 2D rotation
            and translation. "theta" specifies the rotation. "x"
            and "y" specify the translational offset.
        '''
        transformationMatrix = np.array([
            [math.cos(theta), -math.sin(theta), 0, x],
            [math.sin(theta), math.cos(theta), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
            ])
        return transformationMatrix

    def update_joint_coords(self):
        '''update_joint_coords()
            Recompute x and y coordinates of each joint and end effector.
        '''
        
        # "T" is a cumulative transformation matrix that is the result of
        # the multiplication of all transformation matrices up to and including
        # the ith joint of the for loop.
        T = self.get_transformation_matrix(
            self.thetas[0].item(), self.xRoot, self.yRoot)
        for i in range(len(self.lengths) - 1):
            T_next = self.get_transformation_matrix(
                self.thetas[i+1], self.lengths[i], 0)
            T = T.dot(T_next)
            self.joints[:,[i+1]] = T.dot(np.array([[0,0,0,1]]).T)

        # Update end effector coordinates.
        endEffectorCoords = np.array([[self.lengths[-1],0,0,1]]).T
        self.joints[:,[-1]] = T.dot(endEffectorCoords)

    def get_jacobian(self):
        '''get_jacobian()
            Return the 3 x N Jacobian for the current set of joint angles.
        '''

        # Define unit vector "k-hat" pointing along the Z axis.
        kUnitVec = np.array([[0,0,1]], dtype=np.float)

        jacobian = np.zeros((3, len(self.joints[0,:]) - 1), dtype=np.float)
        endEffectorCoords = self.joints[:3,[-1]]

        # Utilize cross product to compute each row of the Jacobian matrix.
        for i in range(len(self.joints[0,:]) - 1):
            currentJointCoords = self.joints[:3,[i]]
            jacobian[:,i] = np.cross(
                kUnitVec, (endEffectorCoords - currentJointCoords).reshape(3,))
        return jacobian

    def update_theta(self, deltaTheta):
        self.thetas += deltaTheta.flatten()
