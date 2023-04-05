# get histogram, fit gaussian.

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
from configDzero_pp import *
from newfunc import *


# Other parameters to pass: 
#--------------------------
# Should find a better way to pass as in a function
#--------------------------------------------------
isRef = 0
fUseRefl = isRef

datafile = '/home/jackbauer/Work/alice/analysis/pp5TeV/D0jet/outData/trial_437.root'
print(datafile)

#signalExtraction.h
#------------------

#------------------
jetmin, jetmax = 0, 50
#print(fptbinsDA)

# 2 ways to plot histogram: 1-ROOT, 2-Python/Matplotlib

#def side_bands(data):
#       print (data)
#       return data
#
#side_bands(5)
#-------------------------------
# Essential functions defined
#-------------------------------
def rawJetSpectra(outdir, prod):
        return 0

def SetReflection(fitter, fLeftFitRange, fRightFitRange, RS, iBin):
        return 0
def SetReflection(fitter, fLeftFitRange, fRightFitRange, RS, ptmin, ptmax):
        return 0
def saveSpectraPlots(fitter, fLeftFitRange, fRightFitRange, RS, iBin):
        return 0
def saveFitParams(outdir, prod):
        return 0
def setHistoDetails(h, scale, color, Mstyle, size):
        width = 2
        if(scale):
                h.Scale(1,"width")
        h.SetMarkerStyle(Mstyle)
        h.SetMarkerColor(color)
        h.SetMarkerSize(size)
        h.SetLineColor(color)
        h.SetLineWidth(width)
        h.SetTitle(0)
        h.GetXaxis().SetTitle(Form("p_{T,%s}(GeV/c)",fDmesonS.Data()))

def SaveCanvas(c, name="tmp"):
        c.SaveAs('%s_pTD%d.png' %(name, fptbinsDA[0]))
        c.SaveAs('%s_pTD%d.pdf' %(name, fptbinsDA[0]))
#------------------------------
# Fitting function python
#------------------------------
def fitter(x, a, b, c):
        return a + b*x + c*x*x
#-------------------------------
# Reading the file
#-------------------------------
File = rt.TFile(datafile,"read")
Dir=File.Get("DmesonsForJetCorrelations")
for i in xrange(0,4):
        #print i
        histlist=Dir.Get("histosD0MBN"+str(i))
        sparse=histlist.FindObject("hsDphiz")
        sparse.GetAxis(0).SetRangeUser(-2.0,2.0)
        sparse.GetAxis(1).SetRangeUser(0,50)
        if(i==0):
                hInvMassptD=sparse.Projection(3,1,2)
                #hInvD=rnp.hist2array(sparse.Projection(3,1,2))
        else:
                hInvMassptD.Add(sparse.Projection(3,1,2))
                #hInvD+=rnp.hist2array(sparse.Projection(3,1,2))

                	
# now the real game begins.
#--------------------------
print(massfit)
print(type(hmass))
print(type(hInvMassptD))
hInvMass, edges = rnp.hist2array(hInvMassptD, return_edges=True)
masshistD = np.zeros([hInvMass.shape[0],hInvMass.shape[2]])
#masshist2Dreb = np.zeros([hInvMass.shape[0], fptbinsDN])
#print[hInvMass.shape[0], fptbinsDN]
for i in xrange(hInvMass.shape[0]): #0th dimension, mass
        for j in xrange(hInvMass.shape[2]): #2nd dimension, D pt
                masshistD[i,j] = sum(hInvMass[i,:,j])
#        masshist1D[i] = masshistD[i,j]

masshist2Dreb = np.zeros([hInvMass.shape[0], fptbinsDN])
masshist2Dreb = rebin2D(masshistD, edges[2], fptbinsDA)

masscenter2D = bincenter(edges[0])
masscenter1D = rebinN(masscenter2D,fRebinMass)

for i in xrange(fptbinsDN):
    plt.plot(masscenter2D, masshist2Dreb[:,i])

masshist1Dreb = []
for i in xrange(fptbinsDN):
	masshist1Dreb.append( rebinN(masshist2Dreb[:,i],fRebinMass) )
masshist1Dreb = np.array(masshist1Dreb)

print(masshist1Dreb.shape)
print(len(masshist2Dreb[0]))
xnx, xny = 3, 5
fig, axs = plt.subplots(xnx, xny)
#plt.subplots_adjust(hspace=0.5)
#axs = axs.ravel()
item = 0

for i in xrange(xnx):
    for j in xrange(xny):
        if (item+j<fptbinsDN):
            print(item+j)
            #axs[i][j].scatter(masscenter2D, masshist2Dreb[:,item+j], marker='+')
            axs[i][j].scatter(masscenter1D, masshist1Dreb[item+j], marker='+')
    item += xny
plt.show()

for i in xrange(0,fptbinsDN):
	pass


help()
