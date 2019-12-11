

class Building:
    """Contains all the parameters used to model the interior of a vapor
    intrusion impacted building.
    Specifically it contains the dimensions of the building, the crack, and
    the amount of sorbing indoor material (if applicable).
    Used in the various simulations.
    """
    def __init__(self, Ae=0.5, w_ck=1e-2, xyz=(10, 10, 3)):
        self.Ae = Ae # air exchange per hour
        self.w_ck = w_ck # crack width in m
        self.xyz = xyz # dimensions of the basement

        self.set_interior_volume()
        self.set_interior_surface_area()
        self.set_crack_area()
        return

    def get_interior_dimensions(self):
        return self.xyz

    def set_interior_surface_area(self):
        x, y, z = self.get_interior_dimensions()

        A_floor = x * y  # area of the floor/ceilung
        A_wall_y = y * z  # area of one type of wall
        A_wall_x = x * z  # area of the other type of wall
        self.A_room = 2 * (A_floor + A_wall_y + A_wall_x)
        return

    def set_interior_volume(self):
        x, y, z = self.get_interior_dimensions()
        self.V = x * y * z
        return

    def set_crack_area(self):
        w_ck = self.get_crack_width()
        x, y ,z = self.get_interior_dimensions()
        self.A_ck = 2*w_ck*(x+y)
        return

    def get_crack_area(self):
        return self.A_ck
    def get_crack_width(self):

        return self.w_ck
    def get_air_exchange_rate(self):
        return self.Ae
    def get_building_volume(self):
        return self.V

    def get_room_area(self):
        return self.A_room

    def set_building_dimensions(self):
        x, y, z = self.get_interior_dimensions()

        A_floor = x * y  # area of the floor/ceilung
        A_wall_y = y * z  # area of one type of wall
        A_wall_x = x * z  # area of the other type of wall

        # assigns parameters to class
        self.V = x * y * z  # building/control volume
        # surface area of the room
        self.A_room = 2 * (A_floor + A_wall_y + A_wall_x)
        return
