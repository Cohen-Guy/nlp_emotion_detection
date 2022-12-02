import os
import pandas as pd
from data_io import DataIO

class FlightsReviewsEmotionDetection:

    def __init__(self):
        self.debug_mode = True
        data_io = DataIO(self.debug_mode)
        self.df = data_io.read_data()
        # self.df.sample(frac=0.1)
        # data_io.export_to_csv(self.df)


if __name__ == "__main__":
    flights_reviews_emotion_detection = FlightsReviewsEmotionDetection()