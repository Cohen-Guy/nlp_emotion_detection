import os
import pandas as pd
from openpyxl import load_workbook

class DataIO(object):

    def __init__(self, debug_mode):
        self.debug_mode = debug_mode
        self.excel_data_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'capstone_airline_reviews3.xlsx')
        self.csv_data_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sampled_capstone_airline_reviews3.csv')

    def read_data(self):
        if self.debug_mode:
            df = pd.read_csv(self.csv_data_file_path, header=1)
        else:
            wb = load_workbook(filename=self.excel_data_file_path)
            ws = wb.active
            df = pd.DataFrame(ws.values)
            df.dropna(inplace=True)
            df = df.reset_index(drop=True)
        return df

    def export_to_csv(self, df):
        export_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sampled_capstone_airline_reviews3.csv')
        df.to_csv(export_file_path, index=False)