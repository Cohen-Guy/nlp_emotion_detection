import datetime
import os
import pandas as pd
from data_io import DataIO
from flights_reviews_emotion_detection.globalsContext import GlobalsContextClass
import sweetviz as sv
import dateutil
import re
import goemotions.er_bert_classifier as bert_classifier
import tensorflow as tf
class FlightsReviewsEmotionDetection:

    def __init__(self):
        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.debug_mode = True
        self.globals_context = GlobalsContextClass(time_str)
        self.data_io = DataIO(self.debug_mode)
        # self.define_flags()

    # def define_flags(self):
    #     self.flags = tf.flags
    #
    #     self.CUSTOM_FLAGS = self.flags.FLAGS
    #
    #     ## Required parameters
    #     self.flags.DEFINE_string("emotion_file", os.path.join(os.path.dirname(os.path.dirname(__file__)), 'goemotions', 'data', 'emotions.txt'),
    #                         "File containing a list of emotions.")
    #
    #     self.flags.DEFINE_string(
    #         "data_dir", os.path.join(os.path.dirname(os.path.dirname(__file__)), 'goemotions', 'data'),
    #         "The input data dir. Should contain the .tsv files (or other data files) "
    #         "for the task.")
    #
    #     self.flags.DEFINE_string(
    #         "bert_config_file", os.path.join(os.path.dirname(os.path.dirname(__file__)), 'goemotions', 'bert', 'bert_config.json'),
    #         "The config json file corresponding to the pre-trained BERT model. "
    #         "This specifies the model architecture.")
    #
    #     self.flags.DEFINE_string("vocab_file", None,
    #                         "The vocabulary file that the BERT model was trained on.")
    #
    #     self.flags.DEFINE_string(
    #         "output_dir", None,
    #         "The output directory where the model checkpoints will be written.")
    #
    #     self.flags.DEFINE_string("test_fname", "test.tsv", "The name of the test file.")
    #     self.flags.DEFINE_string("train_fname", "train.tsv",
    #                         "The name of the training file.")
    #     self.flags.DEFINE_string("dev_fname", "dev.tsv", "The name of the dev file.")
    #
    #     self.flags.DEFINE_boolean("multilabel", False,
    #                          "Whether to perform multilabel classification.")
    #
    #     ## Other parameters
    #
    #     self.flags.DEFINE_string(
    #         "init_checkpoint", os.path.join(os.path.dirname(os.path.dirname(__file__)), 'goemotions', 'output', 'model.ckpt-10000'),
    #         "Initial checkpoint (usually from a pre-trained BERT model).")
    #
    #     self.flags.DEFINE_bool(
    #         "do_lower_case", False,
    #         "Whether to lower case the input text. Should be True for uncased "
    #         "models and False for cased models.")
    #
    #     self.flags.DEFINE_integer(
    #         "max_seq_length", 50,
    #         "The maximum total input sequence length after WordPiece tokenization. "
    #         "Sequences longer than this will be truncated, and sequences shorter "
    #         "than this will be padded.")
    #
    #     self.flags.DEFINE_bool("do_train", True,
    #                       "Whether to run training & evaluation on the dev set.")
    #
    #     self.flags.DEFINE_bool(
    #         "calculate_metrics", True,
    #         "Whether to calculate performance metrics on the test set "
    #         "(FLAGS.test_fname must have labels).")
    #
    #     self.flags.DEFINE_bool(
    #         "do_predict", True,
    #         "Whether to run the model in inference mode on the test set.")
    #
    #     self.flags.DEFINE_bool(
    #         "do_export", False,
    #         "Whether to export the model to SavedModel format.")
    #
    #     self.flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")
    #
    #     self.flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
    #
    #     self.flags.DEFINE_float("num_train_epochs", 4.0,
    #                        "Total number of training epochs to perform.")
    #
    #     self.flags.DEFINE_integer("keep_checkpoint_max", 10,
    #                          "Maximum number of checkpoints to store.")
    #
    #     self.flags.DEFINE_float(
    #         "warmup_proportion", 0.1,
    #         "Proportion of training to perform linear learning rate warmup for. "
    #         "E.g., 0.1 = 10% of training.")
    #
    #     self.flags.DEFINE_float("pred_cutoff", 0.05,
    #                        "Cutoff probability for showing top emotions.")
    #
    #     self.flags.DEFINE_float(
    #         "eval_prob_threshold", 0.3,
    #         "Cutoff probability determine which labels are 1 vs 0, when calculating "
    #         "certain evaluation metrics.")
    #
    #     self.flags.DEFINE_string(
    #         "eval_thresholds", "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99",
    #         "Thresholds for evaluating precision, recall and F-1 scores.")
    #
    #     self.flags.DEFINE_float("sentiment", 0,
    #                        "Regularization parameter for sentiment relations.")
    #
    #     self.flags.DEFINE_float("correlation", 0,
    #                        "Regularization parameter for emotion correlations.")
    #
    #     self.flags.DEFINE_integer("save_checkpoints_steps", 500,
    #                          "How often to save the model checkpoint.")
    #
    #     self.flags.DEFINE_integer("save_summary_steps", 100,
    #                          "How often to save model summaries.")
    #
    #     self.flags.DEFINE_integer("iterations_per_loop", 1000,
    #                          "How many steps to make in each estimator call.")
    #
    #     self.flags.DEFINE_integer("eval_steps", None,
    #                          "How many steps to take to go over the eval set.")
    #
    #     self.flags.DEFINE_string("sentiment_file", "goemotions/data/sentiment_dict.json",
    #                         "Dictionary of sentiment categories.")
    #
    #     self.flags.DEFINE_string(
    #         "emotion_correlations", None,
    #         "Dataframe containing emotion correlation values "
    #         "(if FLAGS.correlation != 0)."
    #     )
    #
    #     self.flags.DEFINE_bool(
    #         "transfer_learning", False,
    #         "Whether to perform transfer learning (i.e. replace output layer).")
    #
    #     self.flags.DEFINE_bool("freeze_layers", False, "Whether to freeze BERT layers.")
    #
    #     self.flags.DEFINE_bool(
    #         "add_neutral", False,
    #         "Whether to add a neutral label in addition to the other labels "
    #         "(necessary when neutral is not part of the emotion file).")

    def eda(self):
        df = self.data_io.read_data()
        df = self.feature_selection(df)
        df = self.feature_engineering(df)
        eda_report = sv.analyze(df)
        report_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eda', 'eda_report.html')
        eda_report.show_html(report_file_path)

    def predict(self, df):
        df = bert_classifier.predict_from_input(df)
        return df

    def ml_flow(self):
        df = self.data_io.read_data()
        df = self.feature_selection(df)
        df = self.feature_engineering(df)
        df = self.predict(df)
        pass

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
    flights_reviews_emotion_detection.ml_flow()