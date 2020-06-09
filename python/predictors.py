"""Implementation of a variety of regression predictors."""
import numpy as np
import scipy
import sklearn.metrics
from PIL import Image

class Predictor():
    """Base class for predictors.

    Parameters
    ----------
    zs_train : list
        List of training observations.
    ys_train : list
        List of training measurements.
    online : boolean
        Flag to determine whether prediction should incorporate
        basic memory.
    
    """

    def __init__(self, zs=[], ys=[], online=False):
        """Create a predictor."""
        self.zs_train = zs
        self.ys_train = ys
        if online:
            self.prev_pred = np.zeros_like(ys[0])
        self.online = online

    def add_data(self, zs, ys):
        """Add data to the predictor.

        Parameters
        ----------
        zs : list
            Observations to add.
        ys : list
            Corresponding labels.

        """

        assert len(zs) == len(ys)
        self.zs_train += zs
        self.ys_train += ys

    def pred(self, zs):
        """Prediction function.

        Parameters
        ----------
        zs : list
            New observations.

        Returns
        -------
        preds : list
            Predicted labels.

        """

        preds, _ = self.compute_pred(zs)
        if self.online:
            if np.linalg.norm(self.preds[-1]) < 1e-6:
                print('using prev pred!')
                self.preds[-1] = self.self.prev_pred
            else:
                self.prev_pred = self.preds[-1]
        return preds

class KernelPredictor(Predictor):
    """Nonparametric Nadarya-Watson kernel estimator.
            y_hat = sum_t y_t * Ind(d(z_t, z) < gamma)

    Parameters
    ----------
    distance : str
        The type of distance metric to use. Currently only l2 is implemented.
    gamma: float
        Bandwidth parameter.
    transform : str
        The type of transformation to perform on observations before
        computing the distance.
    
    """
    def __init__(self, zs=[], ys=[], distance='l2', gamma=1,
                 transform='identity', online=False):
        super().__init__(zs=zs, ys=ys, online=False)
        self.zs_train_arr = None
        distance_dict = {'l2': self._default_distance}
        transform_dict = {'identity': self._default_transform,
                          'resnet': self._resnet_transform,
                          'sift': self._sift_transform,
                          'canny': self._canny_transform,
                          'hog': self._hog_transform,
                          'gaussian': self._gaussian_transform}
        self.distance = distance_dict[distance]
        self.transform = transform_dict[transform]
        self.gamma = gamma

    def _default_distance(self, x, y):
        return sklearn.metrics.pairwise.euclidean_distances(x, y)

    def _default_transform(self, zs):
        return np.array(zs).reshape(len(zs), -1)

    def _sift_transform(self, zs):
        import cv2

        n_features = 10
        sift = cv2.xfeatures2d.SIFT_create(n_features)
        vecs = []
        for z in zs:
            if len(z.shape) < 3:
                rgb_arr = 255*(np.array([z]*3).T)
            else:
                rgb_arr = z[:,:,:]
            _, descriptors = sift.detectAndCompute(np.uint8(rgb_arr), None)
            vecs.append(descriptors[:10].flatten())
        return np.array(vecs)

    def _canny_transform(self, zs):
        import cv2

        vecs = []
        for z in zs:
            if len(z.shape) < 3:
                rgb_arr = 255*(np.array([z]*3).T)
            else:
                rgb_arr = z[:,:,:]
            edges = cv2.Canny(np.uint8(rgb_arr),100,200)
            vecs.append(edges.flatten())
        return np.array(vecs)

    def _gaussian_transform(self, zs):
        import skimage

        vecs = []
        for z in zs:
            if len(z.shape) < 3:
                rgb_arr = 255*(np.array([z]*3).T)
            else:
                rgb_arr = z[:,:,:]
            transform = skimage.filters.gaussian(rgb_arr, sigma=2)
            vecs.append(transform.flatten())
        return np.array(vecs)

    def _hog_transform(self, zs):
        from skimage.feature import hog

        vecs = []
        for z in zs:
            if len(z.shape) < 3:
                rgb_arr = 255*(np.array([z]*3).T)
            else:
                rgb_arr = z[:,:,:]
            _, hog_img = hog(rgb_arr,orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
            vecs.append(hog_img.flatten())
        return np.array(vecs)

    def _resnet_transform(self, zs):
        from img2vec_pytorch import Img2Vec

        img2vec = Img2Vec(cuda=False)
        img_list = []; vecs = []
        for z in zs:
            if len(z.shape) < 3:
                rgb_arr = 255*(np.array([z]*3).T)
            else:
                rgb_arr = z
            img_list.append(Image.fromarray(np.uint8(rgb_arr)))
            vecs += [img2vec.get_vec(img_list[-1], tensor=False)]
        return np.vstack(vecs)

    def add_data(self, zs, ys):
        super().add_data(zs=zs, ys=ys)
        self.zs_train_arr = None

    def compute_pred(self, zs, param_list=None):
        """Compute predictions.

        Parameters
        ----------
        zs : list
            New observations.
        param_list: list, optional
            List of alternate hyperparameters to try.

        Returns
        -------
        preds : list or dict
            Predicted measurements.
        sTs : list or dict
            Coverage terms.

        """
        if self.zs_train_arr is None: # lazy updates
            self.zs_train_arr = self.transform(self.zs_train)
        zs = self.transform(zs)
        distances = self.distance(np.array(self.zs_train_arr), zs)
        if param_list is None:
            preds = []
            sTs = []
            full_mask = distances < self.gamma
            sTs = np.sum(full_mask, axis=0)
            for sT, mask in zip(sTs, full_mask.T):
                if sT == 0: 
                    preds.append(np.zeros_like(self.ys_train[0]))
                else:
                    preds.append(np.sum(np.array(self.ys_train)[mask], axis=0) / sT)
        else:
            all_preds = {}
            all_sTs = {}
            for gamma in param_list:
                preds = []
                sTs = []
                full_mask = distances < gamma
                sTs = np.sum(full_mask, axis=0)
                for sT, mask in zip(sTs, full_mask.T):
                    if sT == 0: 
                        preds.append(np.zeros_like(self.ys_train[0]))
                    else:
                        preds.append(np.sum(np.array(self.ys_train)[mask], axis=0) / sT)
                all_preds[gamma] = preds
                all_sTs[gamma] = sTs
            preds = all_preds
            sTs = all_sTs
        
        return preds, sTs


class KernelRidgePredictor(Predictor):
    """Kernel ridge regression estimator.

    Parameters
    ----------
    lam : float
        Regularization parameter.
    kernel: str
        Type of kernel function.

    """
    
    def __init__(self, lam=1, ys=[], zs=[],
                 kernel='rbf'):
        super().__init__(zs=zs, ys=ys)
        self.lam = lam
        self.gamma = 1e-9
        kernel_dict = {'rbf': self._default_kernel}
        self.kernel = kernel_dict[kernel]
        self.trained = False
        self.K = None

    def _default_kernel(self, zs, ys=None):
        znew = None if ys is None else [y.flatten() for y in ys]
        kernel = sklearn.metrics.pairwise.rbf_kernel([z.flatten() for z in zs], znew, 
                                                    gamma=self.gamma).T
        return kernel

    def add_data(self, zs, ys):
        super().__init__(zs=zs, ys=ys)
        self.trained = False
        self.K = None

    def train(self):
        ys = np.array(self.ys_train)
        zs = np.array(self.zs_train)
        if self.K is None:
            self.K = self.kernel(zs)
        T, _ = self.K.shape
        sv_sq, U = scipy.linalg.eigh(self.K)
        sv_sq[(sv_sq<0)] = 0
        self.coeff = ys.T @ U @ np.diag(1 / (sv_sq + self.lam) ) @ U.T
        self.trained = True

    def compute_pred(self, zs, param_list=None):
        if param_list is None:
            if not self.trained:
                self.train()
            preds = self.kernel(self.zs_train, zs) @ self.coeff.T
        else:
            preds = {}
            kernel_paired = self.kernel(self.zs_train, zs)
            for lam in param_list:
                self.lam = lam
                self.train()
                preds[lam] = kernel_paired @ self.coeff.T
        return preds, None

class FeatureRidgePredictor(Predictor):
    """ Ridge regression estimator.

    Parameters
    ----------
    lam : float
        Regularization parameter.
    features: str
        Type of feature function.

    """

    def __init__(self, lam=1, ys=[], zs=[],
                 features='identity'):
        super().__init__(zs=zs, ys=ys)
        self.lam = lam
        feature_dict = {'identity': self._default_features,
                        'hog': self._hog_features,
                        'canny': self._canny_features}
        self.features = feature_dict[features]
        self.phis = None

    def _default_features(self, zs):
        return np.array([z.flatten() for z in zs])

    def _hog_features(self, zs):
        from skimage.feature import hog

        vecs = []
        for z in zs:
            if len(z.shape) < 3:
                rgb_arr = 255*(np.array([z]*3).T)
            else:
                rgb_arr = z[:,:,:]
            _, hog_img = hog(rgb_arr,orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
            vecs.append(hog_img.flatten())
        return np.array(vecs)

    def _canny_features(self, zs):
        import cv2

        vecs = []
        for z in zs:
            if len(z.shape) < 3:
                # the dot
                rgb_arr = 255*(np.array([z]*3).T)
            else:
                # already an image
                rgb_arr = z[:,:,:]
            edges = cv2.Canny(np.uint8(rgb_arr),100,200)
            vecs.append(edges.flatten())
        return np.array(vecs)

    def add_data(self, zs, ys):
        super().__init__(zs=zs, ys=ys)
        self.trained = False
        self.phis = None

    def train(self):
        ys = np.array(self.ys_train)
        if self.phis is None:
            zs = np.array(self.zs_train)
            self.phis = self.features(zs)
        U, s, VT = scipy.linalg.svd(self.phis, full_matrices=False)
        sv_sq = s**2
        D_sigma = s / (sv_sq + self.lam)
        self.ahat = ys.T @ U @ np.diag(D_sigma) @ VT        
        self.trained = True

    def compute_pred(self, zs, param_list=None):
        zs_features = self.features(zs)
        if param_list is None:
            if not self.trained:
                self.train()
            preds = zs_features @ self.ahat.T
        else:
            preds = {}
            for lam in param_list:
                self.lam = lam
                self.train()
                preds[lam] = zs_features @ self.ahat.T
        return preds, None
