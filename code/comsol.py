from data import Data
import pandas as pd

class COMSOL(Data):
    """
    Class to load, process, and return the COMSOL simulation data
    """

    def __init__(self, file):
        Data.__init__(self, file)
        self.process_raw_data()
        self.A_ck = 0.1 * 4
        return

    def get_crack_area(self):
        return self.A_ck

    def get_raw_data(self):
        return pd.read_csv(self.get_path(), header=4)

    def get_renaming_scheme(self):
        renaming = {'% Time (h)': 'time', 'p_in (Pa)': 'p_in', 'alpha (1)': 'alpha',
                    'c_in (ug/m^3)': 'c_in', 'j_ck (ug/(m^2*s))': 'j_ck', 'm_ads (g)': 'm_ads',
                    'c_ads (mol/kg)': 'c_ads',
                    'c_ads/c_gas (1)': 'c_ads/c_gas', 'alpha_ck (1)': 'alpha_ck',
                    'n_ck (ug/s)': 'n_ck', 'Pe (1)': 'Pe', 'c_gas (ug/m^3)': 'c_gas',
                    'u_ck (cm/h)': 'u_ck', '% K_ads (m^3/kg)': 'K_ads','K_ads (m^3/kg)': 'K_ads',
                    't (h)': 'time', 'c_ads_vol (ug/m^3)': 'c_ads_vol', 'c_liq (ug/m^3)': 'c_liq',
                    '% Pressurization cycles index': 'p_cycle',
                    'Pressurization cycles index': 'p_cycle', 'Q_ck (L/h)': 'Q_ck',
                    '% matsw.comp1.sw1': 'soil', '% Switch 1 index': 'soil',
                    }
        return renaming

    def process_raw_data(self):
        raw_df = self.get_raw_data()
        # removes the druplicate columns from read_csv
        for col in list(raw_df):
            if '.1' in col:
                raw_df.drop(columns=col, inplace=True)

        self.data = raw_df.rename(columns=self.get_renaming_scheme())

        return

    def get_time_data(self):
        df = self.get_data()
        return df['time'].values

    def get_entry_flux_data(self):
        df = self.get_data()
        return df['j_ck'].values

    def get_entry_rate_data(self):
        df = self.get_data()
        return df['n_ck'].values

    def get_concentration_data(self):
        df = self.get_data()
        return df['c_in'].values
    def get_groudwater_concentration(self):
        return self.c_gw
    def set_groundwater_concentration(self):
        df = self.get_data()
        self.c_gw = df['c_in'].values[0]/df['alpha'].values[0]
        return
