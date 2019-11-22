
class Contaminant:
    def __init__(self, contaminant):

        self.contaminant = contaminant
        self.set_molar_mass()
        return

    def set_molar_mass(self):
        contaminant = self.get_contaminant()
        # molar masses are in (g/mol)
        M = {'TCE': 131.38}
        self.M = M[contaminant]
        return

    def get_contaminant(self):
        return self.contaminant

    def get_molar_mass(self):
        return self.M
