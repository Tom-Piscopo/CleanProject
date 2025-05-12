import numpy as np
import pandas as pd

class FEMBeam:
    def __init__(self, length, area, E, rho, xi, nel, ipt=1, bcs="simply"):
        self.length = length
        self.area = area
        self.E = E
        self.rho = rho
        self.xi = xi
        self.nel = nel
        self.ipt = ipt
        self.nnel = 2
        self.ndof = 2
        self.nnode = (self.nnel-1)*(nel+1)
        self.sdof = self.nnode*self.ndof
        self.bcs = bcs

        self.eLength = self.length/self.nel
        self.systemK = np.zeros((self.sdof, self.sdof))
        self.index = np.zeros((self.nnel*self.ndof, 1))

    def feeldof1(self, iel):
        edof = self.nnel*self.ndof
        start = iel*(self.nnel-1)*self.ndof

        for i in range(edof):
            self.index[i,0] = start+i

    def febeam1(self, YoungsModulus):
        c = YoungsModulus * self.xi / (self.eLength ** 3)
        k = c * np.array([
            [12, 6 * self.eLength, -12, 6 * self.eLength],
            [6 * self.eLength, 4 * self.eLength ** 2, -6 * self.eLength, 2 * self.eLength ** 2],
            [-12, -6 * self.eLength, 12, -6 * self.eLength],
            [6 * self.eLength, 2 * self.eLength ** 2, -6 * self.eLength, 4 * self.eLength ** 2]
        ])
        if self.ipt == 1:
            mm = self.rho * self.area * self.eLength / 420
            m = mm * np.array([
                [156, 22 * self.eLength, 54, -13 * self.eLength],
                [22 * self.eLength, 4 * self.eLength**2, 13 * self.eLength, -3 * self.eLength**2],
                [54, 13 * self.eLength, 156, -22 * self.eLength],
                [-13 * self.eLength, -3 * self.eLength**2, -22 * self.eLength, 4 * self.eLength**2]
            ])
        return k, m

    def feasmbl1(self, k):
        edof = np.size(self.index)
        for i in range(edof):
            ii = int(self.index[i,0])
            for j in range(edof):
                jj = int(self.index[j,0])
                self.systemK[ii,jj] = self.systemK[ii,jj] + k[i,j]

    def feaplyc2(self, ff):
        if self.bcs == "simply":
            bcdof = np.array([0, self.sdof-2])
            bcval = np.array([0,0])

            for i in range(np.size(bcdof)):
                c = bcdof[i]
                for j in range(self.sdof):
                    self.systemK[c,j] = 0

                self.systemK[c,c] = 1
                ff[c,0] = bcval[i]
        return ff

if __name__=='__main__':
    length = 1
    area = 1
    UndamagedYoungsModulus = 70e+9
    rho = 1
    xi = 1e-6
    nel = 24
    ipt=1
    bcs="simply"

    beam = FEMBeam(length, area, UndamagedYoungsModulus, rho, xi, nel, ipt=ipt, bcs=bcs)
    ff = np.zeros((beam.sdof,1))

    for iel in range(nel):
        beam.feeldof1(iel)
        k, _ = beam.febeam1(UndamagedYoungsModulus)
        beam.feasmbl1(k)
        beam.feaplyc2(ff)

    print(beam.systemK[1, 2])
    print(beam.systemK[0,0])
    pass

