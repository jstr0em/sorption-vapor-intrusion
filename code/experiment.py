from pandas import read_csv
from data import Data

class Experiment(Data):
    def __init__(self, file):
        Data.__init__(self, file)
        self.set_raw_data()
        return

    def set_raw_data(self):
        path = self.get_path()
        self.data = read_csv(path)
        return
