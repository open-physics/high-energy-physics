import numpy as np
import scipy as sp

def rebin2D(hist, oldedges, newedges):
        # hist and newhist are 2D histogram numpy arrays
        # edges are D pt bin edges of old histogram hist
        # newedges are D pt edges of the new histogram newhist
        # failsafe: edges[0] <= newedges[0]
        newhist = np.zeros([len(hist), len(newedges)-1])
        for i in xrange(len(newedges)):
                for j in xrange(len(oldedges)):
                        if (newedges[i] == oldedges[j]):
                                newhist[:,i] = hist[:,j]
                        while (i+1 < len(newedges) and newedges[i+1] > oldedges[j+1]):
                                newhist[:,i] += hist[:,j+1]
                                j += 1
                        break

        return newhist


def rebin1D(hist, oldedges, newedges):
        # hist and newhist are 1D histogram numpy arrays
        # edges are of old histogram hist
        # newedges are of the new histogram newhist
        # failsafe: edges[0] <= newedges[0]
        newhist = np.zeros([len(newedges)-1])
        for i in xrange(len(newedges)):
                for j in xrange(len(oldedges)):
                        if (newedges[i] == oldedges[j]):
                                newhist[i] = hist[j]
                        while (i+1 < len(newedges) and newedges[i+1] > oldedges[j+1]):
                                newhist[i] += hist[j+1]
                                j += 1
                        break

        return newhist

def rebinN(hist, N):
        # hist 1D histogram numpy array
        # failsafe: N is integer
	if float(N).is_integer():
		newhist = np.zeros([int(len(hist)/N)])
	        for i in xrange(len(newhist)):
	        	for j in xrange(N):
				newhist[i] += hist[N*i + j]
	        return newhist
	else:
		return "Rebin factor is not an integer.\n Please enter an integer"

def bincenter(binedges):
	bincenterarray = np.zeros(len(binedges)-1)
	for i in xrange(len(bincenterarray)):
		bincenterarray[i] = (binedges[i]+binedges[i+1])/2.0
	return bincenterarray
