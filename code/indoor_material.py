from building import Building
from material import Material

class IndoorMaterial(Material, Building):
    """Class that stores methods for the indoor material.
    Mainly to calculate the material volume inside a building.
    """

    def __init__(self, material='cinderblock'):
        Building.__init__(self)
        Material.__init__(self, material=material)
        self.set_material_volume()
        return
    def set_material_volume(self):
        """Sets the total volume of material inside the modeled interior of
        the building.
        """
        material = self.get_material()
        A_room = self.get_room_area()
        penetration_depth = self.get_penetration_depth()
        self.V_mat = A_room * penetration_depth
        return

    def get_material_volume(self):
        """Returns the total volume of material inside the modeled interior of
        the building.
        """
        return self.V_mat

    def get_penetration_depth(self):
        """Returns the depth to which the contaminant penetrates the material.
        This is pretty arbritraily chosen.
        """
        material = self.get_material()
        # depth to which contaminant has been adsorbed/penetrated into the material
        penetration_depth = {'cinderblock': 5e-3, 'wood': 1e-3,
                             'drywall': 1e-2, 'carpet': 1e-2, 'paper': 1e-4, 'none': 0}
        return penetration_depth[material]
