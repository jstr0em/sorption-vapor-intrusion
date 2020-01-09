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

        return np.array([[-(k1*Vbar+Ae), k2*Vbar],[k1, -k2]])

    def get_eigenvalues(self):
        A = self.get_ode_system()
        v, w = np.linalg.eig(A)
        return v, w

    def get_initial_values(self):
        k1, k2, K = self.get_reaction_constants()
        c0 = 2 # ug/m^3 indoor air concentrations

        if self.get_material() != 'none':
            c0_sorb = c0*K
        else:
            c0_sorb = 0
        return np.array([c0, c0_sorb])

    def get_material_solution(self, t):
        """Returns the ODE syste solution for when there is sorbing indoor
        materials inside the building.
        """
        v, w = self.get_eigenvalues()
        lambda1, lambda2 = v
        vec1, vec2 = w[:,0], w[:,1]
        c_inis = self.get_initial_values()
        A, B = np.linalg.solve(w, c_inis)

        # reshaping to enable matrix multiplication
        t = t.reshape(1,-1)
        vec1 = vec1.reshape(-1,1)
        vec2 = vec2.reshape(-1,1)
        return A*np.exp(lambda1*t)*vec1+B*np.exp(lambda2*t)*vec2

    def get_no_material_solution(self,t):
        """Returns the solution to the ODE system where a mitigation has been
        turned on and there is no indoor sorption.
        dc/dt = -Ae*c
        c = c0*exp(-Ae*t)
        """
        c_inis = self.get_initial_values()
        c0 = c_inis[0] # only need the indoor concentraiton (1 is sorbed concentration)
        Ae = self.get_air_exchange_rate()

        # reshaping to enable matrix multiplication
        return c_inis.reshape(-1,1)*np.exp(-Ae*t.reshape(1,-1))


    def get_solution(self,t):
        if self.get_material() != 'none':
            return self.get_material_solution(t)
        else:
            return self.get_no_material_solution(t)

    def get_reduction_time(self, reduction=0.5):
        """
        Calculates when the indoor air concentration has decreased by a certain
        factor given some initial condition.
        """
        c0_in, c0_sorb = self.get_initial_values()
        target = c0_in*reduction
        def find_root(t):
            return self.get_solution(t)[0]-target

        tau = fsolve(find_root, x0=0)
        return tau
