# hfjets  
code from scratch for studying heavy flavour jets  

Physics motivation:  
   1. Heavy flavour:  
                   Formed very early in hard scatterings in heavy-ion collisions.  
                   Negligible annihilation rate.  
                   Participate in whole medium evolution.  

   2. Jets:        More direct access to initial parton kinematics.  
   
   3. Gluon splitting:  

Physical observables:  
   x-section  

Variables:  
   1. p<sub>T,jet</sub> for D-jets.  
   2. z<sub>||</sub> = p<sub>D</sub> / p<sub>jet</sub> for D-jets.  
   3. (future) z<sub>g</sub> = p<sub>T1</sub> / (p<sub>T1</sub> + p<sub>T2</sub>) for two subjets _j1_ and _j2_ tagged with heavy flavours or not.  

Meaningful/final observables:  
   1. R<sub>pA</sub>, R<sub>AA</sub> (nuclear modification factors) = N(pA,AA) / T N(pp)  
   2. R(p<sub>T,jet</sub><sup>ch</sup>, z<sub>||</sub><sup>ch</sup>) = N <sub>D-jet</sub> (p<sub>T,jet</sub><sup>ch</sup>, z<sub>||</sub><sup>ch</sup>)/ N<sub>jet</sub>(p<sub>T,jet</sub><sup>ch</sup>)  

The analysis strategy  proceeds by the following procedure:  
   1. signal extraction by         a) sideband subtraction (default), and  
                                   b) direct jet p<sub>T</sub> extraction (cross-check)  
   2. correction by        a) reconstruction efficiency,  
                           b) b-feed down subtraction,  
                           c) unfolding:   Bayesian  
                                           SVD  
