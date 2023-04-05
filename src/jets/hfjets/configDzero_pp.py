#!/usr/bin/env python #not necessary
# python header file:
import numpy as np
import ROOT as RT

#configDzero_pp.h
#----------------
fptbinsDN = 12
fptbinsDA = np.array([2,3,4,5,6,7,8,10,12,16,20,24,36])

fRebinMass = 2
minf, maxf = 1.71, 2.1

fDmesonS = 'D^0'
fDmesonSpecie = 0
fbkgtype = 0
fsignaltype = 0

fDmass, fDsigma = 1.864, 0.010
fsigmaSignal = 2
fsigmaBkg = [-9, -4, 4, 9]

#---------------
jetplotmin, jetplotmax = 2, 50
#massfit = RT.TF1()
hmass = []
hmass_l = []
hmass_u = []
hmass_c = []
massfit = []
fullfit = []
bkgfit = []
bkgRfit = []
hjetpt = []
hjetpt_sb = []
hjetptsub = []
