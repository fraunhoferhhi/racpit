# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:59:19 2018

@author: Rodrigo Hernangomez
"""

import numpy as np

constants = {'c': 3e8, 'f0': 60e9}

eps = 10.0 ** (- 100)


def range_axis(bw, n_samples, c=constants['c']):
    """
    Calculate range resolution and scope.
    :param bw: FMCW bandwidth in Hz.
    :param n_samples: Number of samples per chirp.
    :param c: Speed of light in m/s.
    :return: Range resolution dr and scope rmax
    """
    dr = c / (2 * bw)
    rmax = dr * n_samples / 2
    return dr, rmax


def doppler_axis(prt, n_chirps, lambda0=constants['c']/constants['f0']):
    """
    Calculate velocity resolution and scope.
    :param prt: Pulse Repetition Time in seconds.
    :param n_chirps: Number of chirps per frame.
    :param lambda0: Wavelength of the radar in meters. For 60 Ghz, this equals around 5 mm.
    :return: Velocity resolution dv and scope vmax
    """
    vmax = lambda0 / (4 * prt)
    dv = 2 * vmax / n_chirps
    return dv, vmax


def rx_mask(rx1, rx2, rx3):
    mask = 0b0
    for i, rx in enumerate([rx1, rx2, rx3]):
        mask += rx << i
    return mask


def normalize_db(db_data, axis=None):
    return db_data - np.max(db_data, axis=axis, keepdims=True)


def absolute(rdm, normalize=False):
    rdm_abs = np.abs(rdm)
    if normalize:
        rdm_abs /= rdm_abs.max()
    return rdm_abs


def mag2db(rdm, normalize=True):
    rdm_ = np.abs(rdm)
    rdm_db = 20 * np.log10(rdm_ + eps)
    if normalize:
        rdm_db = normalize_db(rdm_db)
    return rdm_db


def db2mag(db):
    return np.power(10, db / 20)


def complex2vector(array_cx):
    return np.stack([array_cx.real, array_cx.imag], axis=-1)


def vector2complex(array_vec):
    cx_vec = array_vec.astype(np.complex)
    return cx_vec[..., 0] + cx_vec[..., 1] * 1j


def affine_transform(x, a, b):
    return a * x + b
