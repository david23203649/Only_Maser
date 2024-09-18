import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from tqdm import tqdm
import time
import sympy as sym
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot
import csv
from datetime import datetime
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import time

class Pulses():
    def drive(self, t, args):
        phi = args.get('phi', 0)
        return 1j * args['amp'] * 2 * np.cos(args['omega'] * t + phi) * args['func'](t, args)

    def gaussian(self, t, args):
        t0 = args['t0']
        sigma = args['sigma']
        nsig = args.get('nsig', 1.33)
        return np.exp(-0.5 * ((t - t0 - sigma * nsig / 2) / sigma) ** 2)

    def smoothbox(self, t, args):
        t0 = args['t0']
        width = args['width']
        k = args.get('k', 0.5)
        b = args.get('b', 3)
        return 0.5 * (np.tanh(k * (t - t0) - b) - np.tanh(k * (t - t0 - width) + b))

    def box(self, t, args):
        t0 = args['t0']
        width = args['width']
        return np.heaviside(t - t0, 0) - np.heaviside(t - t0 - width, 0)

    def QM(self, t, args):
        phi = args.get('phi', 0)
        return np.exp(1j * (args['omega_m'] - args['omega_m']) * t + phi)

    def QM_dag(self, t, args):
        phi = args.get('phi', 0)
        return np.exp(1j * (-args['omega_m'] + args['omega_m']) * t + phi)

    def SNAIL3(self, t, args):
        phi = args.get('phi', 0)
        return args['amp'] * np.exp(-1j * (args['omega_s'] + args['omega_m'] - args['omega']) * t + phi) * (
                    2 * args['omega'] * args['func'](t, args) / (args['omega'] ** 2 - args['omega_s'] ** 2))

    def SNAIL3_dag(self, t, args):
        phi = args.get('phi', 0)
        return args['amp'] * np.exp(1j * (args['omega_s'] + args['omega_m'] - args['omega']) * t + phi) * (
                    2 * args['omega'] * args['func'](t, args) / (args['omega'] ** 2 - args['omega_s'] ** 2))

    def SNAIL4(self, t, args):
        phi = args.get('phi', 0)
        return args['amp'] * np.exp(1j * (args['omega_s'] - args['omega_m'] - args['omega']) * t + phi) * (
                    2 * args['omega'] * args['func'](t, args) / (args['omega'] ** 2 - args['omega_s'] ** 2))

    def SNAIL4_dag(self, t, args):
        phi = args.get('phi', 0)
        return args['amp'] * np.exp(-1j * (args['omega_s'] - args['omega_m'] - args['omega']) * t + phi) * (
                    2 * args['omega'] * args['func'](t, args) / (args['omega'] ** 2 - args['omega_s'] ** 2))

    def SNAIL5(self, t, args):
        phi = args.get('phi', 0)
        return np.exp(1j * (args['omega_s'] - args['omega_m']) * t)

    def SNAIL5_dag(self, t, args):
        phi = args.get('phi', 0)
        return np.exp(-1j * (args['omega_s'] - args['omega_m']) * t)

    def qm_from_g4(self, t, args):
        phi = args.get('phi', 0)
        return (args['amp']  * (2 * args['omega'] * args['func'](t, args) / (args['omega'] ** 2 - args['omega_s'] ** 2))) **2


    def cos_drive(self, t, args):
        return (args['amp'] * args['func'](t, args) * np.cos(args['omega'] * t))


    def eta_eta(self,args):
        return (args['amp'] * (
                    2 * args['omega'] * args['func'](t, args) / (args['omega'] ** 2 - args['omega_s'] ** 2))) ** 2
