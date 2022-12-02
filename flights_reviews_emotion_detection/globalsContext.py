import os
import time
import pandas as pd

class GlobalsContextClass:

    def __init__(self, timestr):
        self.timestr = timestr
        self.debug_flag = True
        self.set_column_definitions()

    def set_column_definitions(self):
        self.cols_dict = {
            'boolean_cols':
                [
                    {
                        'field_name': 'recommended',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                ],
            'ordinal_cols':
                [
                    {
                        'field_name': 'overall',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'seat_comfort',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'cabin_service',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'food_bev',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'entertainment',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'ground_service',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'value_for_money',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                ],
            'categorical_cols':
                [
                    {
                        'field_name': 'airline',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'author',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'aircraft',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'traveller_type',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'cabin',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'source',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'destination',
                        'description': '',
                        'exclude_feature_from_training': True,
                        'include_in_correlation': True,
                    },
                ],
            'numerical_cols':
                [

                ],
            'datetime_cols':
                [
                    {
                        'field_name': 'review_date',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'date_flown',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                ],
            'special_handling_cols':
                [
                    {
                        'field_name': 'customer_review',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                    {
                        'field_name': 'route',
                        'description': '',
                        'exclude_feature_from_training': False,
                        'include_in_correlation': True,
                    },
                ],
            'target_col':
                {
                    'field_name': 'DelayBucket',
                    'description': '',
                    'exclude_feature_from_training': False,
                    'include_in_correlation': True,
                },
        }