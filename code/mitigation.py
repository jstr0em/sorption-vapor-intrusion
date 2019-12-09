import numpy as np
import matplotlib.pyplot as plt

from comsol import COMSOL
from material import Material
from contaminant import Contaminant


class Mitigation:
    def __init__(self):
        self.Ae = 0.5 # per hour
        self.k1 = 0.04 # paper
        self.k2 = 88.37 # paper
        self.K = self.k1/self.k2
        self.Vmat = 0.032 # material volume m^3
        self.Vbldg = 300 # building volume m^3

        return

    def get_kinetics(self):
        return self.k1, self.k2

    def get_equilibrium_constant(self):
        return self.K

    def get_air_exchange_rate(self):
        return self.Ae

    def get_building_volume(self):
        return self.Vbldg

    def get_material_volume(self):
        return self.Vmat

    def get_ode_system(self):
        """
        Returns the "mitigation" ODE system governing the desorbing indoor
        material after a mitigation system has been turned on. (n_ck = 0)

        d(c_in)/dt = -(k2*Vbar+Ae)*c_in+k1*Vbar*c_sorb
        d(c_sorb)/dt = k2*c_in-k1*c_sorb
        """
        Vmat = self.get_material_volume()
        Vbldg = self.get_building_volume()
        k1, k2 = self.get_kinetics()
        Ae = self.get_air_exchange_rate()
        Vbar = Vmat/Vbldg # just simpler to write

        return np.matrix([[-(k2*Vbar+Ae), k1*Vbar],[k2, -k1]])

    def get_eigenvalues(self):
        A = self.get_ode_system()
        v, w = np.linalg.eig(A)
        return v, w

    def get_initial_values(self):
        K = self.get_equilibrium_constant()
        c0 = 2 # ug/m^3 indoor air concentrations
        return np.array([c0, c0/K])

    def get_solution(self,t):
        v, w = self.get_eigenvalues()
        lambda1, lambda2 = v
        vec1, vec2 = w[:,0], w[:,1]
        c_inis = self.get_initial_values()
        A, B = np.linalg.solve(w, c_inis)
        return A*np.exp(lambda1*t)*vec1+B*np.exp(lambda2*t)*vec2


x = Mitigation()


times = np.linspace(0,100)

c_in = []
c_sorb = []

for time in times:
    y = x.get_solution(time)
    y = np.array(y)
    print(y)
    c_in.append(y[0])
    c_sorb.append(y[1])


plt.semilogy(times, c_in)
plt.semilogy(times, c_sorb)

plt.show()
