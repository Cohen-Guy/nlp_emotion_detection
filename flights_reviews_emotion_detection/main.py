import datetime
import os
import pandas as pd
from data_io import DataIO
from flights_reviews_emotion_detection.globalsContext import GlobalsContextClass
import sweetviz as sv
import dateutil
import re
# import flights_reviews_emotion_detection.pyplutchik as pyplutchik
from GoEmotionsPytorch.main_goemotions import EmotionDetectionClassification
import plotly.graph_objects as go
class FlightsReviewsEmotionDetection:

    def __init__(self):
        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.debug_mode = False
        self.globals_context = GlobalsContextClass(time_str)
        self.data_io = DataIO(self.debug_mode)

    def eda(self):
        df = self.data_io.read_data()
        df = self.feature_selection(df)
        df = self.feature_engineering(df)
        eda_report = sv.analyze(df)
        report_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eda', 'eda_report.html')
        eda_report.show_html(report_file_path)

    def predict(self, df):
        df = self.emotionDetectionClassification.predict(df)
        return df

    def ml_flow(self):
        self.emotionDetectionClassification = EmotionDetectionClassification()
        df = self.data_io.read_data()
        df = self.feature_selection(df)
        df = self.feature_engineering(df)
        df = self.predict(df)
        pass

    def filter_df(self, df, filter_dict, filter_type):
        for key, value in filter_dict.items():
            if filter_type == 'equal':
                if value is not None:
                    df = df[df[key] == value]
                else:
                    df = df[df[key].isnull()]
            elif filter_type == 'equal-list':
                df = df[df[key].isin(value)]
            elif filter_type == 'larger':
                df = df[df[key] >= value]
            elif filter_type == 'smaller':
                df = df[df[key] <= value]
        return df
    # def airline_emotion_detecion_insights(self):
    #     df = self.data_io.read_preorocessed_data()
    #     df_AA = self.filter_df(df, {'airline': 'American Airlines'}, 'equal')
    #     emotion_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'GoEmotionsPytorch', 'data', 'emotions.txt')
    #     with open(emotion_file_path, "r") as f:
    #         all_emotions = f.read().splitlines()
    #         idx2emotion = {i: e for i, e in enumerate(all_emotions)}
    #     df_emotions_AA = pd.DataFrame(columns=['emotion', 'avg'])
    #     for emotion in all_emotions:
    #         emotion_dict = {'emotion': emotion, 'avg': df_AA[emotion].mean()}
    #         df_emotions_AA = df_emotions_AA.append(emotion_dict, ignore_index=True)
    #     df_emotions_AA = df_emotions_AA.sort_values(by=['avg'], ascending=False)
    #     df_emotions_AA = df_emotions_AA.head(10)
    #     df_SA = self.filter_df(df, {'airline': 'Spirit Airlines'}, 'equal')
    #     df_emotions_SA = pd.DataFrame(columns=['emotion', 'avg'])
    #     for emotion in all_emotions:
    #         emotion_dict = {'emotion': emotion, 'avg': df_SA[emotion].mean()}
    #         df_emotions_SA = df_emotions_SA.append(emotion_dict, ignore_index=True)
    #     df_emotions_SA = df_emotions_SA.sort_values(by=['avg'], ascending=False)
    #     df_emotions_SA = df_emotions_SA.head(10)
    #
    #     fig = go.Figure()
    #     fig.add_trace(go.Bar(
    #         x=df_emotions_AA['emotion'],
    #         y=df_emotions_AA['avg'],
    #         name='Customer Emotions for American Airlines',
    #         marker=dict(color='#0099ff') # blue
    #     ))
    #     fig.add_trace(go.Bar(
    #         x=df_emotions_SA['emotion'],
    #         y=df_emotions_SA['avg'],
    #         name='Customer Emotions for Spirit Airlines',
    #         marker=dict(color='#ffcc66') #yellow
    #     ))
    #     fig.update_layout(barmode='group', xaxis_tickangle=-45,
    #                       font=dict(
    #                           family="Georgia",
    #                           size=30,
    #                           color="RebeccaPurple"),
    #                       yaxis=dict(
    #                         tickformat='.0%'
    #                       )
    #                       )
    #     fig.show()

    def airline_emotion_detecion_insights(self):
        df = self.data_io.read_preorocessed_data()
        emotion_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'GoEmotionsPytorch', 'data', 'emotions.txt')
        with open(emotion_file_path, "r") as f:
            all_emotions = f.read().splitlines()
            idx2emotion = {i: e for i, e in enumerate(all_emotions)}
        # vc = df['dst'].value_counts().sort_values(ascending=False)
        airline_list = ['American Airlines', 'Spirit Airlines', 'British Airways', 'Cathay Pacific Airways', 'Emirates', 'Lufthansa']
        df_airline = self.filter_df(df, {'dst': airline_list}, 'equal-list')
        df_airline_emotions = pd.DataFrame(columns=['dst',
                                                'emotion_1',
                                                'avg_emotion_1',
                                                'emotion_2',
                                                'avg_emotion_2',
                                                'emotion_3',
                                                'avg_emotion_3',
                                                'emotion_4',
                                                'avg_emotion_4'])

        emotions_list = ['admiration', 'fear', 'surprise', 'disgust', 'disappointment', 'gratitude']
        for airline in airline_list:
            airline_dict = {'airline': airline,
                        'emotion_1': emotions_list[0],
                        f"avg_emotion_1": self.filter_df(df, {'airline': airline}, 'equal')[emotions_list[0]].mean(),
                        'emotion_2': emotions_list[1],
                        f"avg_emotion_2": self.filter_df(df, {'airline': airline}, 'equal')[emotions_list[1]].mean(),
                        'emotion_3': emotions_list[2],
                        f"avg_emotion_3": self.filter_df(df, {'airline': airline}, 'equal')[emotions_list[2]].mean(),
                        'emotion_4': emotions_list[3],
                        f"avg_emotion_4": self.filter_df(df, {'airline': airline}, 'equal')[emotions_list[3]].mean(),
                        'emotion_5': emotions_list[3],
                        f"avg_emotion_5": self.filter_df(df, {'airline': airline}, 'equal')[emotions_list[4]].mean(),
                        'emotion_6': emotions_list[3],
                        f"avg_emotion_6": self.filter_df(df, {'airline': airline}, 'equal')[emotions_list[5]].mean(),
                }
            df_airline_emotions = df_airline_emotions.append(airline_dict, ignore_index=True)

        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                #emotions_list = ['admiration', 'fear', 'surprise', 'disgust', 'disappointment', 'gratitude']
                go.Bar(name='Admiration', x=airline_list, y=df_airline_emotions['avg_emotion_1'], yaxis='y', offsetgroup=1),
                go.Bar(name='Fear', x=airline_list, y=df_airline_emotions['avg_emotion_2'], yaxis='y2', offsetgroup=2),
                go.Bar(name='Surprise', x=airline_list, y=df_airline_emotions['avg_emotion_3'], yaxis='y3', offsetgroup=3),
                go.Bar(name='Disgust', x=airline_list, y=df_airline_emotions['avg_emotion_4'], yaxis='y4', offsetgroup=4),
                go.Bar(name='Disappointment', x=airline_list, y=df_airline_emotions['avg_emotion_5'], yaxis='y5', offsetgroup=5),
                go.Bar(name='Gratitude', x=airline_list, y=df_airline_emotions['avg_emotion_6'], yaxis='y6', offsetgroup=6)
            ],
            layout={
                'yaxis': {'title': 'Emotion bases on airline'},
                'yaxis2': {'overlaying': 'y', 'visible': False},
                'yaxis3': {'overlaying': 'y', 'visible': False},
                'yaxis4': {'overlaying': 'y', 'visible': False},
                'yaxis5': {'overlaying': 'y', 'visible': False},
                'yaxis6': {'overlaying': 'y', 'visible': False}
            }
        )
        fig.update_layout(barmode='group', xaxis_tickangle=-45,
                  font=dict(
                      family="Georgia",
                      size=30,
                      color="RebeccaPurple"),
                  yaxis=dict(
                    tickformat='.0%'
                  )
                  )
        fig.show()

    def aircraft_emotion_detecion_insights(self):
        df = self.data_io.read_preorocessed_data()
        df_A320 = self.filter_df(df, {'aircraft_0': 'A320', 'aircraft_1': None, 'aircraft_2': None}, 'equal')
        emotion_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'GoEmotionsPytorch', 'data', 'emotions.txt')
        with open(emotion_file_path, "r") as f:
            all_emotions = f.read().splitlines()
            idx2emotion = {i: e for i, e in enumerate(all_emotions)}
        df_emotions_A320 = pd.DataFrame(columns=['emotion', 'avg'])
        for emotion in all_emotions:
            emotion_dict = {'emotion': emotion, 'avg': df_A320[emotion].mean()}
            df_emotions_A320 = df_emotions_A320.append(emotion_dict, ignore_index=True)
        df_emotions_A320 = df_emotions_A320.sort_values(by=['avg'], ascending=False)
        df_emotions_A320 = df_emotions_A320.head(10)
        df_777 = self.filter_df(df, {'aircraft_0': '777', 'aircraft_1': None, 'aircraft_2': None}, 'equal')
        df_emotions_777 = pd.DataFrame(columns=['emotion', 'avg'])
        for emotion in all_emotions:
            emotion_dict = {'emotion': emotion, 'avg': df_777[emotion].mean()}
            df_emotions_777 = df_emotions_777.append(emotion_dict, ignore_index=True)
        df_emotions_777 = df_emotions_777.sort_values(by=['avg'], ascending=False)
        df_emotions_777 = df_emotions_777.head(10)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_emotions_A320['emotion'],
            y=df_emotions_A320['avg'],
            name='Customer Emotions for Aircraft A320',
            marker=dict(color='#0099ff') # blue # blue 
        ))
        fig.add_trace(go.Bar(
            x=df_emotions_777['emotion'],
            y=df_emotions_777['avg'],
            name='Customer Emotions for Aircraft 777',
            marker=dict(color='#ffcc66') #yellow #yellow
        ))
        fig.update_layout(barmode='group', xaxis_tickangle=-45,
                  font=dict(
                      family="Georgia",
                      size=30,
                      color="RebeccaPurple"),
                  yaxis=dict(
                    tickformat='.0%'
                  )
                  )
        fig.show()

    def dst_emotion_detecion_insights(self):
        df = self.data_io.read_preorocessed_data()
        emotion_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'GoEmotionsPytorch', 'data', 'emotions.txt')
        with open(emotion_file_path, "r") as f:
            all_emotions = f.read().splitlines()
            idx2emotion = {i: e for i, e in enumerate(all_emotions)}
        # vc = df['dst'].value_counts().sort_values(ascending=False)
        dst_list = ['Amsterdam', 'Toronto', 'Los Angeles', 'Paris', 'London', 'Singapore', 'New York', 'Tel Aviv']
        df_dst = self.filter_df(df, {'dst': dst_list}, 'equal-list')
        df_dst_emotions = pd.DataFrame(columns=['dst',
                                                'emotion_1',
                                                'avg_emotion_1',
                                                'emotion_2',
                                                'avg_emotion_2',
                                                'emotion_3',
                                                'avg_emotion_3',
                                                'emotion_4',
                                                'avg_emotion_4'])

        emotions_list = ['admiration', 'fear', 'surprise', 'disgust', 'disappointment', 'gratitude']
        for dst in dst_list:
            dst_dict = {'dst': dst,
                        'emotion_1': emotions_list[0],
                        f"avg_emotion_1": self.filter_df(df, {'dst': dst}, 'equal')[emotions_list[0]].mean(),
                        'emotion_2': emotions_list[1],
                        f"avg_emotion_2": self.filter_df(df, {'dst': dst}, 'equal')[emotions_list[1]].mean(),
                        'emotion_3': emotions_list[2],
                        f"avg_emotion_3": self.filter_df(df, {'dst': dst}, 'equal')[emotions_list[2]].mean(),
                        'emotion_4': emotions_list[3],
                        f"avg_emotion_4": self.filter_df(df, {'dst': dst}, 'equal')[emotions_list[3]].mean(),
                        'emotion_5': emotions_list[3],
                        f"avg_emotion_5": self.filter_df(df, {'dst': dst}, 'equal')[emotions_list[4]].mean(),
                        'emotion_6': emotions_list[3],
                        f"avg_emotion_6": self.filter_df(df, {'dst': dst}, 'equal')[emotions_list[5]].mean(),
                }
            df_dst_emotions = df_dst_emotions.append(dst_dict, ignore_index=True)

        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                #emotions_list = ['admiration', 'fear', 'surprise', 'disgust', 'disappointment', 'gratitude']
                go.Bar(name='Admiration', x=dst_list, y=df_dst_emotions['avg_emotion_1'], yaxis='y', offsetgroup=1),
                go.Bar(name='Fear', x=dst_list, y=df_dst_emotions['avg_emotion_2'], yaxis='y2', offsetgroup=2),
                go.Bar(name='Surprise', x=dst_list, y=df_dst_emotions['avg_emotion_3'], yaxis='y3', offsetgroup=3),
                go.Bar(name='Disgust', x=dst_list, y=df_dst_emotions['avg_emotion_4'], yaxis='y4', offsetgroup=4),
                go.Bar(name='Disappointment', x=dst_list, y=df_dst_emotions['avg_emotion_5'], yaxis='y5', offsetgroup=5),
                go.Bar(name='Gratitude', x=dst_list, y=df_dst_emotions['avg_emotion_6'], yaxis='y6', offsetgroup=6)
            ],
            layout={
                'yaxis': {'title': 'Emotion bases on flight destination'},
                'yaxis2': {'overlaying': 'y', 'visible': False},
                'yaxis3': {'overlaying': 'y', 'visible': False},
                'yaxis4': {'overlaying': 'y', 'visible': False},
                'yaxis5': {'overlaying': 'y', 'visible': False},
                'yaxis6': {'overlaying': 'y', 'visible': False}
            }
        )
        fig.update_layout(barmode='group', xaxis_tickangle=-45,
                  font=dict(
                      family="Georgia",
                      size=30,
                      color="RebeccaPurple"),
                  yaxis=dict(
                    tickformat='.0%'
                  )
                  )
        fig.show()

    # def dst_emotion_detecion_insights(self):
    #     df = self.data_io.read_preorocessed_data()
    #     emotion_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'GoEmotionsPytorch', 'data', 'emotions.txt')
    #     with open(emotion_file_path, "r") as f:
    #         all_emotions = f.read().splitlines()
    #         idx2emotion = {i: e for i, e in enumerate(all_emotions)}
    #     df_London = self.filter_df(df, {'dst': 'London'}, 'equal')
    #     df_emotions_London = pd.DataFrame(columns=['emotion', 'avg'])
    #     for emotion in all_emotions:
    #         emotion_dict = {'emotion': emotion, 'avg': df_London[emotion].mean()}
    #         df_emotions_London = df_emotions_London.append(emotion_dict, ignore_index=True)
    #     df_emotions_London = df_emotions_London.sort_values(by=['avg'], ascending=False)
    #     df_emotions_London = df_emotions_London.head(10)
    #     df_LHR = self.filter_df(df, {'dst': 'LHR'}, 'equal')
    #     df_emotions_LHR = pd.DataFrame(columns=['emotion', 'avg'])
    #     for emotion in all_emotions:
    #         emotion_dict = {'emotion': emotion, 'avg': df_LHR[emotion].mean()}
    #         df_emotions_LHR = df_emotions_LHR.append(emotion_dict, ignore_index=True)
    #     df_emotions_LHR = df_emotions_LHR.sort_values(by=['avg'], ascending=False)
    #     df_emotions_LHR = df_emotions_LHR.head(10)
    #     fig = go.Figure()
    #     fig.add_trace(go.Bar(
    #         x=df_emotions_London['emotion'],
    #         y=df_emotions_London['avg'],
    #         name='Customer Emotions for London Destination',
    #         marker=dict(color='#0099ff') # blue
    #     ))
    #     fig.add_trace(go.Bar(
    #         x=df_emotions_LHR['emotion'],
    #         y=df_emotions_LHR['avg'],
    #         name='Customer Emotions for London Heathrow Destination',
    #         marker=dict(color='#ffcc66') #yellow
    #     ))
    #     fig.update_layout(barmode='group', xaxis_tickangle=-45,
    #               font=dict(
    #                   family="Georgia",
    #                   size=30,
    #                   color="RebeccaPurple"),
    #               yaxis=dict(
    #                 tickformat='.0%'
    #               )
    #               )
    #     fig.show()

    def value_counts_emotion_detecion_insights(self):
        df = self.data_io.read_preorocessed_data()
        emotion_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'GoEmotionsPytorch', 'data', 'emotions.txt')
        with open(emotion_file_path, "r") as f:
            all_emotions = f.read().splitlines()
            idx2emotion = {i: e for i, e in enumerate(all_emotions)}
        df_emotions_count = pd.DataFrame(columns=['emotion', 'count'])
        for emotion in all_emotions:
            emotion_dict = {'emotion': emotion, 'count': self.filter_df(df, {'top_1_emotion': emotion}, 'equal').shape[0]}
            df_emotions_count = df_emotions_count.append(emotion_dict, ignore_index=True)
        df_emotions_count = df_emotions_count.sort_values(by=['count'], ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_emotions_count['emotion'],
            y=df_emotions_count['count'],
            name='Emotions Counts',
            marker=dict(color='#0099ff') # blue
        ))
        fig.update_layout(barmode='group', xaxis_tickangle=-45,
                  font=dict(
                      family="Georgia",
                      size=30,
                      color="RebeccaPurple"),
                  )
        fig.show()
    def overall_emotion_detecion_insights(self):
        df = self.data_io.read_preorocessed_data()
        emotion_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'GoEmotionsPytorch', 'data', 'emotions.txt')
        with open(emotion_file_path, "r") as f:
            all_emotions = f.read().splitlines()
            idx2emotion = {i: e for i, e in enumerate(all_emotions)}
        df_high = self.filter_df(df, {'overall': 9}, 'larger')
        df_emotions_top = pd.DataFrame(columns=['emotion', 'avg'])
        for emotion in all_emotions:
            emotion_dict = {'emotion': emotion, 'avg': df_high[emotion].mean()}
            df_emotions_top = df_emotions_top.append(emotion_dict, ignore_index=True)
        df_emotions_top = df_emotions_top.sort_values(by=['avg'], ascending=False)
        df_emotions_top = df_emotions_top.head(10)
        df_low =self.filter_df(df, {'overall': 2}, 'smaller')
        df_emotions_low = pd.DataFrame(columns=['emotion', 'avg'])
        for emotion in all_emotions:
            emotion_dict = {'emotion': emotion, 'avg': df_low[emotion].mean()}
            df_emotions_low = df_emotions_low.append(emotion_dict, ignore_index=True)
        df_emotions_low = df_emotions_low.sort_values(by=['avg'], ascending=False)
        df_emotions_low = df_emotions_low.head(10)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_emotions_top['emotion'],
            y=df_emotions_top['avg'],
            name='High overall score Customer Emotions',
            marker=dict(color='#0099ff') # blue
        ))
        fig.add_trace(go.Bar(
            x=df_emotions_low['emotion'],
            y=df_emotions_low['avg'],
            name='Low overall score Customer Emotions',
            marker=dict(color='#ffcc66') #yellow
        ))
        fig.update_layout(barmode='group', xaxis_tickangle=-45,
                  font=dict(
                      family="Georgia",
                      size=30,
                      color="RebeccaPurple"),
                  yaxis=dict(
                    tickformat='.0%'
                  )
                  )
        fig.show()
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
            aircrafts = str(aircrafts).replace('Boeing ', '')
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

    def convert_date(self, date):
        try:
            return dateutil.parser.parse(date)
        except:
            return date

    def extract_src_and_dst(self, df):
        for index, row in df.iterrows():
            route = row['route']
            if route is not None:
                re_result = re.search('(.*?) to (.*?) (via (.*))|(.*?) to (.*)', route)
                try:
                    groups = re_result.groups()
                    if groups[0] is not None:
                        df.loc[index, 'src'] = groups[0]
                        df.loc[index, 'dst'] = groups[1]
                        df.loc[index, 'via'] = groups[3]
                    else:
                        df.loc[index, 'src'] = groups[4]
                        df.loc[index, 'dst'] = groups[5]
                except:
                    pass


    def determine_verified(self, group_0):
        if group_0 is None:
            return None
        elif 'Verified Review' in group_0 or 'Trip Verified' in group_0:
            return True
        elif 'Not Verified' in group_0:
            return False
        else:
            return None
    def process_review(self, df):
        for index, row in df.iterrows():
            customer_review = row['customer_review']
            re_result = re.search('((âœ…)? Verified Review \||(âœ…)? Trip Verified \||(âœ…)?Not Verified \|)?(.*)', customer_review)
            try:
                self.determine_verified(re_result.groups()[0])
            except:
                pass
            df.loc[index, 'verified'] = self.determine_verified(re_result.groups()[0])
            df.loc[index, 'review_text'] = re_result.groups()[len(re_result.groups())-1]

    def feature_engineering(self, df):
        df['review_date'] = df['review_date'].apply(self.convert_date)
        # if self.debug_mode:
        df['date_flown'] = df['date_flown'].apply(self.convert_date)
        df['year_flown'] = df['date_flown'].dt.year
        df['month_flown'] = df['date_flown'].dt.month
        self.extract_aircrafts(df)
        self.extract_src_and_dst(df)
        self.process_review(df)
        df = df.drop(['review_date', 'date_flown', 'aircraft', 'route', 'customer_review'], axis=1)
        return df

    def preprocessing(self, df):
        df = self.feature_selection(df)



if __name__ == "__main__":
    flights_reviews_emotion_detection = FlightsReviewsEmotionDetection()
    flights_reviews_emotion_detection.dst_emotion_detecion_insights()