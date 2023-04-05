from ROOT import TObject
import numpy as np

class ParticleTree(TObject):
    #`self` is an instance of the object(TObject) itself.
    #`particle` is the Pythia particle to be passed, basically the first argument
    def __init__(self, particle):
        self.particle = particle
        self.id = self.particle.id()
        self.e  = self.particle.e()
        self.px = self.particle.px()
        self.py = self.particle.py()
        self.pz = self.particle.pz()
        self.m  = self.particle.m()
        self._val = [
                self.id
               ,self.e 
               ,self.px
               ,self.py
               ,self.pz
               ,self.m 
                ]

        self._daughters = []

    def set_val(self,other):
        self._val[0]= other.id
        self._val[1]= other.e 
        self._val[2]= other.px
        self._val[3]= other.py
        self._val[4]= other.pz
        self._val[5]= other.m 
        return np.array(self._val)

    def name_type(self,nametype):
        names = ['id','energy','px','py','pz','mass']
        #nametype=''
        for name in names:
            nametype+=(name+'/D:')
        return nametype[:-1]
    def get_val(self):
        #return np.array([self.id, self.e, self.px, self.py, self.pz, self.m])
        return np.array(self._val)

    def set_daughters(self,daughters_list):
        #self._daughters is the list/array of all daughters and properties in the form of ParticleTree
        #input daughters_list is [[id,e,px...],[id,e,px...]]
        #returned value should be [[id,id...][e,e...][px,px...]]
        ##for p in range(len(daughters_list[0])):
        ##    self._daughters.append(np.array(daughters_list)[:,p])
        self._daughters = []
        daughters_list = np.array(daughters_list)
        for p in range(len(daughters_list[0])):
            #print(daughters_list[:,p])
            self._daughters.append(daughters_list[:,p])
        return np.array(self._daughters)

    def get_daughters(self):
        return np.array(self._daughters)

class FamilyTree(ParticleTree):
    def __init__(self,particle):
        super().__init__(particle)
        
