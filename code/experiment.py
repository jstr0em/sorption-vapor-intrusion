from pandas import read_csv
#from data import Data

class Experiment:
    def __init__(self, file):
        #Data.__init__(self, file)
        self.sorption_data = read_csv(file)
        return

    def get_sorption_data(self):
        return self.sorption_data
