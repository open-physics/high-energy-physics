import numpy as np
import scipy as sp
import matplotlib.pyplot as plt # error due to old numpy version being used
import matplotlib
import ROOT as RT
import rootpy.ROOT as rt
import root_numpy as rnp
import pandas as pd

from rootpy.io import root_open
from root_numpy import root2array, tree2array, hist2array
from root_numpy import testdata

from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as optimization
#from configDzero_pp import fptbinsDN,fptbinsDA,fRebinMass,minf,maxf,fDmesonS,fDmesonSpecie,fbkgtype,fsignaltype fDmass fDsigma
#from configDzero_pp import *


def bkgExpo(x, b1, b2, b3):
	return b1 * np.exp(-b2*x) + b3

def signalGauss(x, a, b, c):
	return a * np.exp((-(x-b)**2)/(2*c**2))

def PyFitter(x, a,b,c, b1,b2,b3):
	return signalGauss(x,a,b,c) + bkgExpo(x,b1,b2,b3)


