# get histogram, fit gaussian.
# Author: Auro Mohanty
# auro.mohanty@cern.ch
# Utrecht University

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
jetmin, jetmax = 0, 50

# 2 ways to plot histogram: 1-ROOT, 2-Python/Matplotlib

#def side_bands(data):
#	print (data)
#	return data
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
	
##Following way of plotting histograms in python
##----------------------------------------------
##hInvMassptD.Draw()
##hInv=rnp.hist2array(hInvMassptD)
##print("3(mass),1(jetpt),2(dpt)")
##hh_full,edges = rnp.hist2array(hInvMassptD.ProjectionX(),return_edges=True)
##binedges = np.array(edges)
##plt.plot(binedges[0][:-1]+(binedges[0][1]-binedges[0][0])/2. ,hh_full,'+') # 1.Updated:midpoints used now 0.Don't use bin[0][:-1]. Rather try using the mid points.
##plt.show()

#
#-------------------------
hmean = RT.TH1D("hmean","hmean",fptbinsDN,fptbinsDA)
hsigma = RT.TH1D("hmean","hmean",fptbinsDN,fptbinsDA)
hrelErr = RT.TH1D("hmean","hmean",fptbinsDN,fptbinsDA)
hsign = RT.TH1D("hmean","hmean",fptbinsDN,fptbinsDA)
hsb = RT.TH1D("hmean","hmean",fptbinsDN,fptbinsDA)
hSignal = RT.TH1D("hmean","hmean",fptbinsDN,fptbinsDA)
hReflRS = RT.TH1D("hmean","hmean",fptbinsDN,fptbinsDA)

xnx, xny = 3, 4
###if(fptbinsDN > 4 and fptbinsDN > 7):
###	xnx, xny = 2, 3
###elif(fptbinsDN > 6 and fptbinsDN > 10):
###	xnx, xny = 3, 3
###elif(fptbinsDN > 9 and fptbinsDN > 13):
###	xnx, xny = 3, 4
###else:
###	xnx, xny = 4, 4
###
c2 = rt.TCanvas("c2", "c2", 1200, 1200)
c2.Divide(xnx, xny)

# now the real game begins.
#--------------------------
for i in xrange(0,fptbinsDN):
	hh = hInvMassptD.ProjectionX(
			'hh_%d' %(i),
			hInvMassptD.GetYaxis().FindBin(jetmin),
			hInvMassptD.GetYaxis().FindBin(jetmax)-1,
			hInvMassptD.GetZaxis().FindBin(fptbinsDA[i]),
			hInvMassptD.GetZaxis().FindBin(fptbinsDA[i+1])-1,
		)
	hh.Rebin(fRebinMass)
	hh.GetXaxis().SetRangeUser(minf,maxf)
	hh.SetTitle('%.1lf < pt^{%s} < %.1lf' %(fptbinsDA[i],fDmesonS,fptbinsDA[i+1]))
	# AliHFInv implementation
	#------------------------
	hmassfit1D = hh.Clone("hmassfit")
	hmassfit = RT.TH1F()
	hmassfit1D.Copy(hmassfit)
	if(fDmesonSpecie):
		hmassfit.SetMaximum(hmassfit.GetMaximum()*1.3)
	hmin = max(minf,hmassfit.GetBinLowEdge(2))
	hmax = min(maxf,hmassfit.GetBinLowEdge(hmassfit.GetNbinsX()))

	fitterp = RT.AliHFInvMassFitter(hmassfit, hmin, hmax, fbkgtype, fsignaltype)
	fitterp.SetInitialGaussianMean(fDmass)
	fitterp.SetInitialGaussianSigma(fDsigma)
	
	#if(fUseRefl and fDmesonSpecie == 0):
		#if(fSystem) SetReflection(fitterp, hmin, hmax, RS, i+firstPtBin) 
			#older way from Fabio's templates for p-Pb
		#else SetReflection(fitterp, hmin, hmax, RS, fptbinsDA[i], fptbinsDA[i+1]) 
			#new for pp, new templates from D-jet code
	
	fitterp.MassFitter(False) #FIT ERROR
	
	h = fitterp.GetHistoClone()
	massfit.append(fitterp.GetMassFunc())
	massfit[i].SetRange(hmin, hmax)
	massfit[i].SetLineColor(4)
	fullfit.append(h.GetFunction("funcmass"))
	if(fullfit[i]):
		fullfit[i].SetName('fullfit_%d' %(i))
	hmass.append(h.Clone('hmass_%d' %(i)))
	hmass[i].SetName('hmass_%d' %(i))
	hmass[i].GetYaxis().SetTitle('Entries')
	hmass[i].GetXaxis().SetTitle('Invariant Mass (GeV/c^{2})')
	bkgfit.append(fitterp.GetBackgroundRecalcFunc())
	bkgfit[i].SetRange(hmin, hmax)
	bkgfit[i].SetLineColor(2)
	bkgfit[i].SetName('bkgFit_%d' %(i))

	# bkg + reflection function
	# -------------------------
#	#if(fUseRefl and fDmesonSpecie == 0):
#		bkgRfit.append(fitterp.GetBkgPlusReflFunc())
#		bkgRfit[i].SetName('bkgFitWRef_%d' %(i))
#		bkgRfit[i].SetRange(hmin,hmax)
#		bkgRfit[i].SetLineColor(15)
#		hReflRS.SetBinContent(i+1, RS)
#		hReflRS.SetBinError(i+1,0)


	pad = c2.GetPad(i+1)
	fitterp.DrawHere(pad,3,0)

	Dsigma, Dmean, DmeanUnc, DsigmaUnc = 0, 0, 0, 0

	if(not fullfit[i]):
		print("====== Fit failed for bin: %d" %(i))

	Dsigma = fitterp.GetSigma()
	DsigmaUnc = fitterp.GetSigmaUncertainty()
	Dmean = fitterp.GetMean()
	DmeanUnc = fitterp.GetMeanUncertainty()

	signal_c_min = Dmean-fsigmaSignal*Dsigma
	signal_c_max = Dmean+fsigmaSignal*Dsigma
	signal_l_min = Dmean+fsigmaBkg[0]*Dsigma
	signal_l_max = Dmean+fsigmaBkg[1]*Dsigma
	signal_u_min = Dmean+fsigmaBkg[2]*Dsigma
	signal_u_max = Dmean+fsigmaBkg[3]*Dsigma
	if(signal_l_min < hmin):
		signal_l_min = hmin
	if(signal_u_max < hmax):
		signal_u_max = hmax
	
	halfbin = hmass[i].GetXaxis().GetBinWidth(1)*0.5
	# signal
	binmin = hmass[i].GetXaxis().FindBin(signal_c_min)
	binmax = hmass[i].GetXaxis().FindBin(signal_c_max)
	min_sig = hmass[i].GetXaxis().GetBinCenter(binmin) - halfbin
	max_sig = hmass[i].GetXaxis().GetBinCenter(binmax-1) + halfbin
	# side-band 1
	binmin_sb1 = hmass[i].GetXaxis().FindBin(signal_l_min)
	binmax_sb1 = hmass[i].GetXaxis().FindBin(signal_l_max)
	min_sb1 = hmass[i].GetXaxis().GetBinCenter(binmin_sb1) - halfbin
	max_sb1 = hmass[i].GetXaxis().GetBinCenter(binmax_sb1-1) + halfbin
	# side-band 2
	binmin_sb2 = hmass[i].GetXaxis().FindBin(signal_u_min)
	binmax_sb2 = hmass[i].GetXaxis().FindBin(signal_u_max)
	min_sb2 = hmass[i].GetXaxis().GetBinCenter(binmin_sb2) - halfbin
	max_sb2 = hmass[i].GetXaxis().GetBinCenter(binmax_sb2-1) + halfbin
	
	s, serr, bkg, bkgerr= RT.Double(0), RT.Double(0), RT.Double(0), RT.Double(0)
	srelerr, bkgref, ref = 0, 0, 0
	sb1, sb2, ref1, ref2 = 0, 0, 0, 0
	fitterp.Signal(fsigmaSignal, s, serr)
	fitterp.Background(min_sig, max_sig, bkg, bkgerr)
	sb1 = bkgfit[i].Integral(min_sb1, max_sb1)/hmass[i].GetBinWidth(1)
	sb2 = bkgfit[i].Integral(min_sb2, max_sb2)/hmass[i].GetBinWidth(1)
	#if(fUseRefl and fDmesonSpecie == 0):
	#	bkgref = bkgRfit[i].Integral(min_sig, max_sig)/hmass[i].GetBinWidth(1)
	#	ref = bkgref - bkg
	#	ref1 = (bkgRfit[i].Integral(min_sb1, max_sb1)/hmass[i].GetBinWidth(1)) - sb1
	#	ref2 = (bkgRfit[i].Integral(min_sb2, max_sb2)/hmass[i].GetBinWidth(1)) - sb2

	signf, signferr = RT.Double(0), RT.Double(0)
	sob, soberr = 0, 0
	fitterp.Significance(fsigmaSignal, signf, signferr)
	if(s):
		srelerr = serr/s
	if(bkg):
		sob = s/bkg 
	else:
		sob = s
	if(bkg and bkgerr):
		soberr = np.sqrt((serr/bkg)**2 + (s/bkg/bkg*bkgerr)**2)
	else:
		soberr = serr
#	Canvas Write statements. To be done later
#	---------------------------------
#	twodigits = True
#	if(soberr*100.0 > 35.0):
#		twodigits = False
#	if(twodigits):
#		pvSig.AddText()
	
	# Fitting results
	# ---------------
	if(fDmesonSpecie):
		hmean.SetBinContent(i+1, Dmean*1000)
		hmean.SetBinError(i+1, DmeanUnc*1000)
	else:
		hmean.SetBinContent(i+1, Dmean)
		hmean.SetBinError(i+1, DmeanUnc)

	hsigma.SetBinContent(i+1, srelerr)
	hsign.SetBinContent(i+1, signf)
	hsign.SetBinError(i+1, signferr)
	hsb.SetBinContent(i+1,sob)
	hsb.SetBinError(i+1,soberr)
	hSignal.SetBinContent(i+1,s)
	hSignal.SetBinError(i+1,serr)
	
	# Side band drawing
	#------------------
	hmass_l.append(hmass[i].Clone("hmass_l"))
	hmass_l[i].GetXaxis().SetRangeUser(signal_l_min,signal_l_max)
	hmass_l[i].SetName("hmass_l_%d" %(i))
	hmass_u.append(hmass[i].Clone("hmass_u"))
	hmass_u[i].GetXaxis().SetRangeUser(signal_u_min,signal_u_max)
	hmass_u[i].SetName("hmass_u_%d" %(i))
	hmass_c.append( hmass[i].Clone("hmass_c") )
	hmass_c[i].GetXaxis().SetRangeUser(signal_c_min,signal_c_max)
	hmass_c[i].SetName("hmass_c_%d" %(i))
	
	hmass_l[i].SetFillColor(rt.kBlue+2)
	hmass_u[i].SetFillColor(rt.kBlue+2)
	hmass_c[i].SetFillColor(rt.kRed+2)
	hmass_l[i].SetFillStyle(3004)
	hmass_u[i].SetFillStyle(3004)
	hmass_c[i].SetFillStyle(3005)
	
	hmass_l[i].Draw("hsame")
	hmass_u[i].Draw("hsame")
	hmass_c[i].Draw("hsame")

	# jet pt spectrum- signal
	#-----------------------
	hjetpt.append(hInvMassptD.ProjectionY(
			('hjetpt_%d' %(i)),
			hInvMassptD.GetXaxis().FindBin(signal_c_min), 
			hInvMassptD.GetXaxis().FindBin(signal_c_max)-1,
			hInvMassptD.GetZaxis().FindBin(fptbinsDA[i]), 
			hInvMassptD.GetZaxis().FindBin(fptbinsDA[i+1])-1))

	hjetpt[i].GetXaxis().SetRangeUser(jetplotmin,jetplotmax)
	hjetpt[i].SetTitle("%.1lf < pt^{%s} < %.1lf" %(fptbinsDA[i],fDmesonS,fptbinsDA[i+1]))
	hjetpt[i].SetMarkerColor(rt.kRed+2)
	hjetpt[i].SetLineColor(rt.kRed+2)

	# jet pt spectrum- sidebands
	#---------------------------
	hjetpt_sb1 = hInvMassptD.ProjectionY(("hjetpt_sb1%d" %(i)),
			hInvMassptD.GetXaxis().FindBin(signal_l_min), 
			hInvMassptD.GetXaxis().FindBin(signal_l_max)-1,
			hInvMassptD.GetZaxis().FindBin(fptbinsDA[i]), 
			hInvMassptD.GetZaxis().FindBin(fptbinsDA[i+1])-1)
	hjetpt_sb1.SetTitle(("%.1lf < pt^{%s} < %.1lf" %(fptbinsDA[i],fDmesonS,fptbinsDA[i+1])))
	hjetpt_sb1.SetMarkerColor(rt.kBlue+2)
	hjetpt_sb1.SetLineColor(rt.kBlue+2)
	hjetpt_sb2 = hInvMassptD.ProjectionY(("hjetpt_sb2%d" %(i)),
			hInvMassptD.GetXaxis().FindBin(signal_u_min), 
			hInvMassptD.GetXaxis().FindBin(signal_u_max)-1,
			hInvMassptD.GetZaxis().FindBin(fptbinsDA[i]), 
			hInvMassptD.GetZaxis().FindBin(fptbinsDA[i+1])-1)
	hjetpt_sb2.SetTitle(("%.1lf < pt^{%s} < %.1lf" %(fptbinsDA[i],fDmesonS,fptbinsDA[i+1])))
	#hjetpt_sb2.SetMarkerColor(kBlue+2)
	#hjetpt_sb2.SetLineColor(kBlue+2)
	hjetpt_sb.append(hjetpt_sb1.Clone(("hjetpt_sb_%d" %(i))))
	hjetpt_sb[i] += hjetpt_sb2

	hjetpt[i].GetXaxis().SetRangeUser(jetplotmin,jetplotmax)
	hjetpt[i].GetXaxis().SetTitle("p_{T,ch jet} (GeV/c)")
	hjetpt[i].GetXaxis().SetTitleOffset(1.)
	hjetpt_sb[i].GetXaxis().SetRangeUser(jetplotmin,jetplotmax)

	# scale sidebands
	#----------------
	scalingB = bkg/(sb1+sb2)
	#if(fUseRefl and fDmesonSpecie == 0):
	#	scalingS = s / (s + ref - ( (ref1 + ref2) * bkg )/(sb1+sb2) )

	hjetpt_sb[i].Scale(scalingB)
	# jet pt spectrum- subtract bkg from signal
	#-----------------------------------------
	hjetptsub.append( hjetpt[i].Clone(("hjetptsub_%d" %(i))))
	hjetptsub[i] -= hjetpt_sb[i]
	#if(fUseRefl and fDmesonSpecie == 0):
	#	hjetptsub[i].Scale(scalingS)
	
	if(fsigmaSignal==2):
		hjetptsub[i].Scale(1./0.9545)
	hjetptsub[i].SetMarkerColor(rt.kGreen+3)
	hjetptsub[i].SetLineColor(rt.kGreen+3)
	help()




help()
