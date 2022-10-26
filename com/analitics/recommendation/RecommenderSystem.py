import abc

import numpy as np
import pandas as pd


from com.analitics.algorithms.FrequentItemSets import PandasAprioriAlgorithm
from com.analitics.recommendation.RecommendationEngines import UserFrequentItemsEngine
from com.analitics.utilities import Utilities, Constants


class RecommenderSystem:
    """
    Abstraction of Recommender System class
    """

    def __init__(self, data_set, transaction_df, date, n_users=10, n_recommendations=10, top=3):
        """
        Constructs a Recommender System process.
        :param data_set: The dataset.
        :param transaction_df: The transaction matrix.
        :param date: Point of reference to generate predictions.
        :param n_users: The number of users to make predictions.
        :param n_recommendations: The number of recommendations to generate.
        :param top: Consider the top-k consumed items by the user.
        """
        self.n_recommendations = n_recommendations
        self.top = top
        self.date = date
        self.n_users = n_users
        self.items = list(np.sort(data_set[Constants.PRODUCT_ID].unique()))
        self.users = list(np.sort(data_set[Constants.ACCOUNT_ID].unique()))
        self.items_string_ids = [str(item_id) for item_id in self.items]
        self.data_set = data_set
        self.transaction_df = transaction_df

        # General information about items consumed by all users in the dataset.
        # In case there isn't enough information to make personalized recommendations.

        self.all_items_history = Utilities.date_filter_dataframe(
            data_set,
            self.date)

        self.items_by_quantity = Utilities.product_group_by(
            self.all_items_history,
            Constants.QUANTITY)

        self.items_quantity_mean = self.items_by_quantity.mean().sort_values(
            by=Constants.QUANTITY,
            ascending=False)

        self.items_by_day_week = Utilities.product_group_by(
            self.all_items_history,
            Constants.DAY_OF_WEEK)

        self.items_day_week_mode = self.items_by_day_week.agg(pd.Series.mode)

        self.test_users = [33237129, 33611229, 33236163, 33238638, 40620450, 33710592,
                           33240375, 37545453, 33259770, 40518873, 33256614]

    @abc.abstractmethod
    def recommend_items(self):
        """
        Generate the recommendations.
        :return:
        """
        pass


class FrequentItemSetRecommenderSystem(RecommenderSystem):
    """
    Defines a Recommendation System using Frequent Item-Sets.
    """
    def __init__(
            self, data_set,
            transaction_df, date, genera_recommendations,
            n_users=10, n_recommendations=10,
            top=3, support=3,
            confidence=0.7, random_users=False):
        """
        Constructs a Recommendation System using Frequent Item-Sets
        :param data_set: The dataset.
        :param transaction_df: The transaction matrix.
        :param date: Point of reference to generate predictions.
        :param n_users: The number of users to make predictions.
        :param n_recommendations: The number of recommendations to generate.
        :param top: Consider the top-k consumed items by the user.
        :param support: The minimum support to generate frequent item-sets
        :param confidence: The minimum confidence for associaton rules.
        """
        super().__init__(data_set, transaction_df, date, n_users, n_recommendations, top)
        self.support = support
        self.confidence = confidence
        self.random_users = random_users
        self.n_users = n_users
        self.n_recommendations = n_recommendations
        self.genera_recommendations = genera_recommendations

    def recommend_items(self):
        """
        Generate the recommendations.
        :return: Recommendations for every user in test.
        """
        users_recommendations = []

        if self.random_users:
            self.test_users = Utilities.random_select(self.users, k=self.n_users)

        for user_id in self.test_users:
            user_recommendations = []
            recommended_items = []
            # User history information of the buys of the user before limit_date
            user_history_data = Utilities.get_user_history_df(
                self.data_set,
                user_id,
                self.date).copy()

            # Transactions of the buys of the user before limit_date
            user_history_transactions = Utilities.get_user_history_df(
                self.transaction_df,
                user_id,
                self.date).copy()

            # If there is a single row, then produce personalized recommendation.
            # If not then consider all users preferences.
            if user_history_transactions.shape[0] > 1:
                user_recommendation, recommended_items = self.run_user_recommendation(
                    user_id,
                    user_history_data,
                    user_history_transactions)

            for general_rec in self.genera_recommendations:
                if general_rec[0] not in recommended_items:
                    rec = (user_id,) + general_rec
                    user_recommendations.append(rec)

            users_recommendations += user_recommendations[:self.n_recommendations]

        return users_recommendations

    def run_user_recommendation(self, user_id, user_history_data, user_history_transactions):
        """
        Performs an Algorithm to generate personalized recommendations for the user.
        :param user_id: The user_id.
        :param user_history_data: The user history dataset.
        :param user_history_transactions: The user history transactions.
        :return: The recommendations as DataFrame.
        """
        # Use Apriori Algorithm
        algorithm = PandasAprioriAlgorithm(
            user_history_transactions,
            self.items_string_ids,
            min_support=self.support,
            min_confidence=self.confidence)

        engine = UserFrequentItemsEngine(
            user_id,
            user_history_data,
            user_history_transactions)
        engine.generate_recommendations(algorithm, k=self.top)

        return (engine.recommendations, engine.recommended_items)

    def run_general_recommendation(self):
        """
        Performs  recommendations based on what all users preferences.
        :return: The recommendations as DataFrame.
        """
        pass
