import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root

from indoor_material import IndoorMaterial
from kinetics import Kinetics
from material import get_all_materials

class Mitigation(IndoorMaterial, Kinetics):
    def __init__(self, material='cinderblock', contaminant='TCE'):
        IndoorMaterial.__init__(self,material=material)
        Kinetics.__init__(self, material=material, contaminant=contaminant)



        return



    def get_ode_system(self):
        """
        Returns the "mitigation" ODE system governing the desorbing indoor
        material after a mitigation system has been turned on. (n_ck = 0)

        d(c_in)/dt = -(k2*Vbar+Ae)*c_in+k1*Vbar*c_sorb
        d(c_sorb)/dt = k2*c_in-k1*c_sorb
        """
        Vmat = self.get_material_volume()
        Vbldg = self.get_building_volume()
        k1, k2, K = self.get_reaction_constants()
        Ae = self.get_air_exchange_rate()
        Vbar = Vmat/Vbldg # just simpler to write

        return np.array([[-(k2*Vbar+Ae), k1*Vbar],[k2, -k1]])

    def get_eigenvalues(self):
        A = self.get_ode_system()
        v, w = np.linalg.eig(A)
        return v, w

    def get_initial_values(self):
        k1, k2, K = self.get_reaction_constants()
        c0 = 2 # ug/m^3 indoor air concentrations
        return np.array([c0, c0/K])

    def get_solution(self,t):
        v, w = self.get_eigenvalues()
        lambda1, lambda2 = v
        vec1, vec2 = w[:,0], w[:,1]
        c_inis = self.get_initial_values()
        A, B = np.linalg.solve(w, c_inis)
        return A*np.exp(lambda1*t)*vec1+B*np.exp(lambda2*t)*vec2

    def get_reduction_time(self, reduction=10):
        """
        Calculates when the indoor air concentration has decreased by a certain
        factor given some initial condition.
        """
        c0_in, c0_sorb = self.get_initial_values()
        target = c0_in/reduction
        def find_root(t):
            return self.get_solution(t)[0]-target

        tau = fsolve(find_root, x0=0)
        return tau
