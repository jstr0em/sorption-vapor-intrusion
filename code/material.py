
def get_all_materials():
    """ Returns a list of all the materials we studied.
    """
    return ['cinderblock', 'paper', 'carpet', 'drywall', 'wood', 'soil', 'none']

class Material:
    def __init__(self, material):
        self.material = material
        self.set_material_density()
        return

    def get_material(self):
        return self.material

    def set_material_density(self):
        material = self.get_material()
        # densities in g/m^3
        density = {'drywall': 0.6e6, 'cinderblock': 2.0e6,
                   'carpet': 1.314e6, 'wood': 0.86e6, 'paper': 0.8e6, 'soil': 1.46e6, 'none': 0}
        # soil is based on sandy loam data
        self.rho = density[material]
        return

    def get_material_density(self):
        return self.rho

    def get_materials(self):
        return ['drywall', 'cinderblock', 'carpet', 'wood', 'paper', 'soil']
