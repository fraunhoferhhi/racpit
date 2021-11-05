import numpy as np
from ifxaion.daq import Daq
import scipy.interpolate
import xarray as xr


def rcsellipsoid(a, b, c, phi,theta):
    rcs = (np.pi * a**2 * b**2 * c**2) / (a**2 * (np.sin(theta))**2 * (np.cos(phi))**2 +
                                          b**2 * (np.sin(theta))**2 * (np.sin(phi))**2 + c**2 * (np.cos(theta))**2)**2
    return rcs


class Segment:
    """ Defines a body segment and allows to calculate RCS of segment """

    def __init__(self, segmentPositions, ellipsoidParams, aspect):
        self.ellipsoidParams = ellipsoidParams
        self.segmentPositions = segmentPositions
        self.aspect = aspect


    def calculateRange(self, radarloc):
        """
        Calculates the distance to the radar for all segment positions
        :return: vector of the distances [d1, d2, d3, ...]
        """
        self.r_dist = np.abs(self.segmentPositions - radarloc)
        self.r_total = np.sqrt(self.r_dist[:, 0] ** 2 + self.r_dist[:, 1] ** 2 + self.r_dist[:, 2] ** 2)
        return self.r_total


    def calculateAngles(self, radarloc):
        """
        Calculates the angles to the radar for all segment positions
        :return: vector of phi [phi1, phi2, phi3,...] and vector of theta [theta1,theta2,theta3,...]
        """
        A = np.column_stack((
            radarloc[0] - self.segmentPositions[:, 0],
            radarloc[1] - self.segmentPositions[:, 1],
            radarloc[2] - self.segmentPositions[:, 2]
        ))

        B = np.column_stack((
            self.aspect[:, 0],
            self.aspect[:, 1],
            self.aspect[:, 2]
        ))

        a_dot_b = np.sum(A * B, axis=1)
        a_sum_sqrt = np.sqrt(np.sum(A * A, axis=1))
        b_sum_sqrt = np.sqrt(np.sum(B * B, axis=1))

        theta = np.arccos(a_dot_b / (a_sum_sqrt * b_sum_sqrt))
        phi = np.arcsin((radarloc[1] - self.segmentPositions[:, 1]) /
                             np.sqrt(self.r_dist[:, 0] ** 2 + self.r_dist[:, 1] ** 2))

        return phi, theta

    def calculateRCS(self, phiAngle, thetaAngle):
        """
        Calculates the RCS
        :return: vector of RCS [rcs1, rcs2, rcs3,...]
        """
        a = self.ellipsoidParams[0]
        b = self.ellipsoidParams[1]
        c = self.ellipsoidParams[2]
        rcs = rcsellipsoid(a, b, c, phiAngle, thetaAngle)
        return rcs


def load(path, verbose=False):
    """
    Read the skeleton data with help of the Daq module.
    """

    daq = Daq(rec_dir=path)

    skeleton_df = daq.skeletons.data

    skeletonTimestamps = skeleton_df.index

    duration = (skeletonTimestamps[-1]-skeletonTimestamps[0]).total_seconds()

    if verbose:
        print("Reading skeleton data: {} skeletons, in {} second recording".format(len(skeleton_df), duration))

    return skeleton_df


def interpolate(skeletons, timestamp_seconds):
    sk_data = np.array(skeletons.Data)
    sk_data = np.stack(sk_data)
    sk_timestamps = skeletons.index

    sk_interpolation = scipy.interpolate.interp1d(sk_timestamps.total_seconds(), sk_data,
                                                  axis=0, fill_value="extrapolate", kind='linear')
    sk_new = sk_interpolation(timestamp_seconds)
    return sk_new


def to_xarray(data, timestamps, name="Skeletons", attrs=None):
    skeleton_da = xr.DataArray(data, dims=("time", "space", "keypoints"),
                               name=name, attrs=attrs,
                               coords={"time": timestamps.to_numpy(),
                                       "space": ["x", "y", "z"],
                                       "keypoints": list(keypoints)}).assign_attrs(units="m")
    return skeleton_da


def get_edges(skeleton):
    if isinstance(skeleton, xr.DataArray):
        sk_data = skeleton.values
    else:
        sk_data = skeleton
    return np.stack([[sk_frame[:, segment] for segment in coco_edges] for sk_frame in sk_data])


keypoints = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle")

# Edges between keypoints used for the COCO API to visualize skeletons
# See https://github.com/facebookresearch/Detectron/issues/640
coco_edges = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
              [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
