import numpy as np
import pandas as pd

from com.analitics.utilities import Constants


class UserFrequentItemsEngine:
    """
    Engine is the actual class that produces recommendations.
    This engine made use Frequent Items and Association Rules to
    create recommendations for a user.
    """

    def __init__(self, user_id, user_history_df, user_transactions_df):
        """
        Creates an UserFrequentItemsEngine instance.
        :param user_id: The user_id.
        :param user_history_df: The user history dataset.
        :param user_transactions_df: The user history transactions.
        """
        self.user_id = user_id                              # User ID to make recommendations
        self.user_history_df = user_history_df              # User History dataset
        self.user_transactions_df = user_transactions_df    # User transactions matrix

        # List of Tuples: (account_id, product_id, recommended quantity, date of week)
        self.recommendations = []

        # Keeps track of the recommended items.
        # No need to process the same item everytime.
        self.recommended_items = []

        # All items consumed by the user.
        items = list(np.sort(user_history_df[Constants.PRODUCT_ID].unique()))
        self.all_items_ids = [str(item_id) for item_id in items]

    def item_quantity_day_prediction(self, item_id):
        """
        Predicts the quantity and day week of the shopping for the given item.
        :param item_id: The item id.
        :return: (quantity, day_week)
        """
        item_history = self.user_history_df[
            (self.user_history_df[Constants.PRODUCT_ID] == item_id)]

        quantity = int(item_history[Constants.QUANTITY].mean())
        day_week = int(item_history[Constants.DAY_OF_WEEK].mode()[0])

        return (quantity, day_week)

    def get_recommendations(self):
        """
        List of tuples that contains the recommendation information.
        :return: List of tuples: (account_id, product_id, recommended quantity, date of week)
        """
        return self.recommendations

    def get_recommendations_df(self):
        """
        Returns Recommendations for the user as DataFrame.
        Columns names are Account_id, Product_id, Recommendation_quantity and Day_of_week
        :return:
        """
        columns = [
            Constants.ACCOUNT_ID,
            Constants.PRODUCT_ID,
            Constants.RECOMMENDED_QUANTITY,
            Constants.DAY_OF_WEEK]

        return pd.DataFrame(self.recommendations, columns=columns)

    def top_k_items(self, k=3):
        """
        Returns the top-k consumed items on average by the user.
        :param k: Integer, the k-most consumed items. Defaults, 3.
        :return: A list of items_ids.
        """
        item_transaction_mean = self.user_transactions_df[self.all_items_ids].mean(axis=0).sort_values(ascending=False)
        return list(item_transaction_mean[:k].index)

    def generate_recommendations(self, algorithm, k=3):
        """
        Generates recommendations given and Algorithm class.
        :param algorithm: An implementation of Algorithm class.
        :param k: Recommends the k-most consumed items by the user.
        Defaults, 3.
        :return:
        """
        algorithm.run()
        associations = algorithm.associations_as_df()

        for most_consumed_item in self.top_k_items(k):
            self.add_item_recommendation(most_consumed_item)
            self.process_association_items(most_consumed_item, associations)

    def add_item_recommendation(self, item):
        """
        Adds the item recommendation information.
        :param item: The item id.
        """
        quantity, day_of_week = self.item_quantity_day_prediction(int(item))

        self.recommendations.append(
            (self.user_id, item, quantity, Constants.DAY_NAMES[day_of_week]))

        self.recommended_items.append(item)

    def process_association_items(self, item, associations_results):
        """
        Process association rules for the given item.
        """
        associated_items = self.search_association_rules(
            associations_results,
            {item})

        for associated_item in associated_items:
            for item in associated_item:
                if item not in self.recommended_items:
                    self.add_item_recommendation(item)

    @staticmethod
    def get_frequent_items(association_df, item_set):
        """
        Gets the Frequent_items column of the associations DataFrame.
        :param association_df: The associations DataFrame.
        :param item_set: The items_set.
        :return: The Frequent_items column.
        """
        return association_df[
            association_df[Constants.CONSUMED_ITEMS] == item_set][Constants.FREQUENT_ITEMS]

    def search_association_rules(self, association_df, search_item_set):
        """
        Searches the association rules for the given.
        :param association_df: The associations rules DataFrame.
        :param search_item_set: The item-set to be searched.
        :return:
        """
        associations = []
        for item_set in association_df[Constants.CONSUMED_ITEMS]:
            if len(item_set) == len(search_item_set) and item_set == search_item_set:
                associations += list(self.get_frequent_items(association_df, item_set))
                break

        return associations
