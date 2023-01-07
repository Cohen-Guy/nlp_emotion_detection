import os
import pandas as pd
from openpyxl import load_workbook
from itertools import islice


class DataIO(object):

    def __init__(self, debug_mode):
        self.debug_mode = debug_mode
        self.excel_data_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'capstone_airline_reviews3.xlsx')
        self.csv_data_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sampled_capstone_airline_reviews3.csv')
        self.preprocessed_data_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'GoEmotionsPytorch', 'output', 'prediction_results.csv')

    def read_data(self):
        if self.debug_mode:
            df = pd.read_csv(self.csv_data_file_path, header=0)
        else:
            wb = load_workbook(filename=self.excel_data_file_path)
            ws = wb.active
            data = ws.values
            cols = next(data)
            df = pd.DataFrame(data, columns=cols)
            df.dropna(subset=['airline', 'review_date', 'date_flown', 'customer_review'], inplace=True)
            df = df.reset_index(drop=True)
        return df

    def read_preorocessed_data(self):
        df = pd.read_csv(self.preprocessed_data_file_path, header=0)
        return df

    def export_to_csv(self, df):
        export_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sampled_capstone_airline_reviews3.csv')
        df.to_csv(export_file_path, index=False)


if __name__ == '__main__':
    dataIO = DataIO(False)
    df = dataIO.read_data()
    df = df.sample(frac=0.01, random_state=42)
    dataIO.export_to_csv(df)
