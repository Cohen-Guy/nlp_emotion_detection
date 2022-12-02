import datetime
import os
import pandas as pd
from data_io import DataIO
from flights_reviews_emotion_detection.globalsContext import GlobalsContextClass
import sweetviz as sv
import dateutil
import re
class FlightsReviewsEmotionDetection:

    def __init__(self):
        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.debug_mode = True
        self.globals_context = GlobalsContextClass(time_str)
        self.data_io = DataIO(self.debug_mode)

    def eda(self):
        df = self.data_io.read_data()
        df = self.feature_selection(df)
        df = self.feature_engineering(df)
        eda_report = sv.analyze(df)
        report_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eda', 'eda_report.html')
        eda_report.show_html(report_file_path)

    def ml_flow(self):
        df = self.data_io.read_data()

    def feature_selection(self, df):
        self.boolean_cols = [boolean_col['field_name'] for boolean_col in self.globals_context.cols_dict['boolean_cols'] if not boolean_col['exclude_feature_from_training']]
        self.ordinal_cols = [ordinal_col['field_name'] for ordinal_col in self.globals_context.cols_dict['ordinal_cols'] if not ordinal_col['exclude_feature_from_training']]
        self.categorical_cols = [categorical_col['field_name'] for categorical_col in self.globals_context.cols_dict['categorical_cols'] if not categorical_col['exclude_feature_from_training']]
        self.numerical_cols = [numerical_col['field_name'] for numerical_col in self.globals_context.cols_dict['numerical_cols'] if not numerical_col['exclude_feature_from_training']]
        self.datetime_cols = [datetime_col['field_name'] for datetime_col in self.globals_context.cols_dict['datetime_cols'] if not datetime_col['exclude_feature_from_training']]
        self.special_handling_cols = [special_handling_col['field_name'] for special_handling_col in self.globals_context.cols_dict['special_handling_cols'] if not special_handling_col['exclude_feature_from_training']]
        self.selected_columns = self.boolean_cols + self.ordinal_cols + self.categorical_cols + self.numerical_cols + self.datetime_cols + self.special_handling_cols
        return df[self.selected_columns]

    def extract_aircrafts(self, df):
        for index, row in df.iterrows():
            aircrafts = row['aircraft']
            aircrafts = aircrafts.replace('Boeing ', '')
            if ',' in aircrafts:
                aircraft_list = aircrafts.split(',')
            elif '&' in aircrafts:
                aircraft_list = aircrafts.split('&')
            elif 'and' in aircrafts:
                aircraft_list = aircrafts.split('and')
            elif '+' in aircrafts:
                aircraft_list = aircrafts.split('+')
            else:
                aircraft_list = aircrafts.split('/')
            df.loc[index, 'aircraft_0'] = aircraft_list[0]
            if (len(aircraft_list) > 1):
                df.loc[index, 'aircraft_1'] = aircraft_list[1]
            if (len(aircraft_list) > 2):
                df.loc[index, 'aircraft_2'] = aircraft_list[2]

    def convert_date(self, date_str):
        return dateutil.parser.parse(date_str)

    def extract_src_and_dst(self, df):
        for index, row in df.iterrows():
            route = row['route']
            re_result = re.search('(.*?) to (.*)?( via )(.*)', route)
            df.loc[index, 'source'] = re_result.group(1)
            df.loc[index, 'dst'] = re_result.group(2)
            df.loc[index, 'connection'] = re_result.group(3)
    def feature_engineering(self, df):
        df['review_date'] = df['review_date'].apply(self.convert_date)
        self.extract_aircrafts(df)
        self.extract_src_and_dst(df)
        pass

    def preprocessing(self, df):
        df = self.feature_selection(df)



if __name__ == "__main__":
    flights_reviews_emotion_detection = FlightsReviewsEmotionDetection()
    flights_reviews_emotion_detection.eda()