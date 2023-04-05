# main01.py is a part of the PYTHIA event generator.
# Copyright (C) 2019 Torbjorn Sjostrand.
# PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
#
# This is a simple test program. It fits on one slide in a talk.  It
# studies the charged multiplicity distribution at the LHC. To set the
# path to the Pythia 8 Python interface do either (in a shell prompt):
#      export PYTHONPATH=$(PREFIX_LIB):$PYTHONPATH
# or the following which sets the path from within Python.
#
# Use "python-config --include" to find the include directory and
# then configure Pythia ""--with-python-include=*".

from ParticleTree import *
import sys
import pythia8
from ROOT import TH1D, TTree, TFile
import ROOT
import numpy as np
from array import array

################
pythia = pythia8.Pythia()
pythia.readFile("configX3872.cmnd");
pythia.init()
################
#nEvents=1000000000
nEvents=10
################
fX3872 = TFile('x3872.root','RECREATE')
tX3872 = TTree('TreeX3872', 'TreeX3872')

part_list = [10441,20443,445,9920443,9940923]
partTree = ParticleTree(pythia8.Particle()) # particle of class ParticleTree created.[initiation]
tX3872.Branch('partTree',partTree.get_val(),partTree.name_type(''))

prt_dauTree = ParticleTree(pythia8.Particle()) # particle of class ParticleTree created.[initiation]

print('ok')
# Begin event loop. Generate event. Skip if error. List first one.
for iEvent in range(0, nEvents):
    while not pythia.next(): continue
    for prt in pythia.event:
        if prt.id() in part_list:
            partTree.set_val(ParticleTree(prt))
            if(len(prt.daughterList())):
                dau_list = []
                for daughters in prt.daughterList():
                    prt_dau = pythia.event[daughters]
                    prt_dauTree.set_val(ParticleTree(prt_dau))
                    dau_list.append(prt_dauTree.get_val())
                partTree.set_daughters(dau_list)
                print(partTree.get_daughters()[0])
                print('=====================================================')
                print('')
            tX3872.Fill()

# End of event loop. Statistics. Histogram. Done.
pythia.stat();

fX3872.Write()
fX3872.Close()
