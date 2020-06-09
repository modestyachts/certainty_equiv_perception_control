"""Implementation of ORB-SLAM based predictors and helper functions."""
import orbslam2
import numpy as np
import time as datetime
import experiments
import tqdm.autonotebook
import os


def get_transform(true_data, slam_data):
    true_position, true_times = true_data
    slam_position, slam_times = slam_data

    true_position = np.array(true_position)
    slam_position = np.array(slam_position)
    true_times = np.array(true_times)
    slam_data = np.array(slam_data)

    ## temporal alignment
    time_indices = [np.argmin(np.abs(true_times - t)) for t in slam_times]
    v = true_position[time_indices].T
    u = slam_position.T

    u_sim, rot, trans = sim_transform(u, v)
    u_align, w = affine_regress(u_sim, v)

    a = w[:,0][:,None]
    b = w[:,1][:,None]

    def transform(pts):
        # argument is n by 3
        # return is n by 3
        pts = np.asarray(pts).reshape((-1, 3)).T
        return ((rot.dot(pts) + trans) * a + b).T

    # todo: use transform to compute u_align

    max_err = np.max(np.abs(v[:2] - u_align[:2]))
    print("transform computed, max error %.3f" % max_err)

    return transform

def sim_transform(u, v):
    # Horn's method
    # make sure u and v are 3xn arrays
    u = u.reshape((3, -1))
    v = v.reshape((3, -1))
    u_mean = u.mean(axis=1, keepdims=True)
    v_mean = v.mean(axis=1, keepdims=True)
    u_centered = u - u_mean
    v_centered = v - v_mean

    W = np.zeros((3, 3))
    for j in range(u.shape[1]):
        W += np.outer(u_centered[:,j], v_centered[:,j])
    U, d, Vh = np.linalg.svd(W.T)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2,2] = -1
    rot = U.dot(S).dot(Vh)
    trans = v_mean - rot.dot(u_mean)
    u_align = rot.dot(u) + trans
    return u_align, rot, trans

def affine_regress(u, v):
    # transform u to v by a 1d affine transform
    assert u.shape[0] == v.shape[0]
    ws = []; aligned_us = []
    for i in range(u.shape[0]):
        X = np.array([u[i], np.ones_like(u[i])]).T
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(v[i])
        ws.append(w)
        u_align = X.dot(w)
        aligned_us.append(u_align)
    return np.array(aligned_us), np.array(ws)

class OrbPredictor():
    """ORB-SLAM based predictor.

    Parameters
    ----------
    fps : float
        Frames per second.
    scale : float
        Scale between state-space and CARLA coordinates.
    height : float
        Fixed height for all observations.
    online : boolean
        Flag for whether operation is online.
    settings_path : str
        Path to camera parameter file.
    vocab_path : str
        Path to ORB vocabulary file.
    
    """

    def __init__(self, fps=10., scale=20., height=40.,
                 online=False, settings_path='carla.yaml',
                 vocab_path='~/ORB_SLAM2/Vocabulary/ORBvoc.txt'):
        self.slam = orbslam2.System(os.path.expanduser(vocab_path),
                                    settings_path, orbslam2.Sensor.MONOCULAR)
        self.slam.set_use_viewer(True)
        self.slam.initialize()
        self.fps = fps
        self.scale = scale
        self.height = height
        self.prev_line = None
        self.online = online
        self.reset_between = False

    def get_xyz(self, ys_train):
        """Get 3d CARLA position from measurements.""" 
        xypos = self.scale * np.array(ys_train)
        heights = self.height * np.ones([xypos.shape[0],1])
        xyz_true = np.hstack([xypos, heights])
        return xyz_true

    def process_train(self, zs_train, ys_train, cutoff=None):
        """Construct database of training images and fit transform.""" 
        self.num_train_images = cutoff if cutoff is not None else len(zs_train)
        # initialize blank frame
        self.blank_image = np.zeros_like(zs_train[0]).astype(np.uint8)
        xyz_true = self.get_xyz(ys_train)
        true_times = []
        for idx in range(self.num_train_images): 
            image = zs_train[idx].astype(np.uint8)
            self.tframe = idx / self.fps
            self.slam.process_image_mono(image, self.tframe)
            datetime.sleep(1./self.fps)
            true_times.append(self.tframe)

        trajectory = self.slam.get_trajectory_points()
        slam_times = []; slam_positions = []
        for line in trajectory:
            time, R, t = self.read_pos(line)
            if time is None:
                time = slam_times[-1]
                t = slam_positions[-1]
            slam_times.append(time)
            slam_positions.append(t)

        self.coord_transform = get_transform((xyz_true, true_times),
                                        (slam_positions, slam_times))

    def process_grid(self, zs_grid, speed=False, no_mapping=True):
        """Prediction function.

        Parameters
        ----------
        zs_grid : list
            Observations along grid.
        speed : boolean
            Flag for whether to speed evaluation by turning off pauses.
        no_mapping : boolean
            Flag for whether to prevent mapping. 

        Returns
        -------
        pos_est : array
            Predicted positions.

        """
        num_grid_images = len(zs_grid)
        pos_est = [0] * num_grid_images
        # visit in a random order so filtering has no effect.
        if no_mapping:
            permutation = np.random.permutation(num_grid_images)
        else:
            permutation = np.arange(num_grid_images)
        for idx in tqdm.autonotebook.tqdm(range(num_grid_images)):
            # clear tracking with blank frame.
            if no_mapping: 
                tframe = (self.num_train_images + idx - 0.5) / self.fps
                self.slam.process_image_mono(self.blank_image, tframe)
            if not speed: 
                datetime.sleep(0.1/self.fps)
            tframe = (self.num_train_images + idx) / self.fps
            image = zs_grid[permutation[idx]].astype(np.uint8)
            self.slam.process_image_mono(image, tframe)
            if not speed: 
                datetime.sleep(0.9/self.fps)
            _, R, t = self.read_pos(self.slam.get_trajectory_points()[-1])
            if t is not None:
                p_est = self.coord_transform(t).flatten()
            else:
                p_est = np.zeros(3)
            pos_est[permutation[idx]] = p_est
        return np.array(pos_est).reshape(-1, 3)

    def read_pos(self, res):
        """Parse pose from ORB-SLAM output.""" 
        if self.prev_line is not None:
            if res == self.prev_line and not self.online:
                return None, None, None
        self.prev_line = res
        time, r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3 = res
        R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
        t = np.array([t1, t2, t3])
        return time, R, t


    def pred(self, z, tframe=None, reset_between=False):
        """Predict from single image [z].""" 
        if tframe is None: 
             self.tframe += 1/self.fps
             tframe = self.tframe
        if reset_between or self.reset_between:
            self.slam.process_image_mono(self.blank_image, tframe-0.5/self.fps)
        image = z[0].astype(np.uint8)
        self.slam.process_image_mono(image, tframe)
        _, R, t = self.read_pos(self.slam.get_trajectory_points()[-1])
        if t is not None:
            pos_est = self.coord_transform(t)
        else:
            pos_est = np.zeros(3)
        return [pos_est.flatten()[:2] / self.scale]

    def shutdown(self):
        self.slam.shutdown()


