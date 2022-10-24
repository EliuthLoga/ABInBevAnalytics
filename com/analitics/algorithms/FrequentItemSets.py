import abc
import pandas as pd

from abc import ABC
from com.analitics.utilities import Utilities, Constants

"""
Module containing Recommender Systems Algorithms.
"""


class Algorithm:
    """
    Abstracts Algorithm class.
    """

    @abc.abstractmethod
    def run(self):
        """
        Runs the algorithm.
        """
        pass

    @abc.abstractmethod
    def results(self):
        """
        Returns the Results of the algorithm
        :return: Results.
        """
        pass


class FrequentItemSetAlgorithm(Algorithm):
    """
    Defines a Frequent Item-Set algorithm.
    It constructs the Association Rules.
    """
    def results(self):
        """
        Returns the Association Rules found by the Algorithm.
        :return: List of Tuples (item_set, association, confidence)
        """
        return self.association_rules()

    @abc.abstractmethod
    def association_rules(self):
        """
        Returns the association rules.
        :return: List of Tuples (item_set, association, confidence)
        """
        pass

    @abc.abstractmethod
    def frequent_item_set_as_df(self):
        """
        Returns frequent item-sets as DataFrame.
        The columns are item_set and support
        :return: DataFrame.
        """
        pass

    @abc.abstractmethod
    def associations_as_df(self):
        """
        Returns the Association as DataFrame.
        The columns are item_set, association and confidence
        :return: DataFrame
        """
        pass


class AprioriAlgorithm(FrequentItemSetAlgorithm):
    """
    Abstracts the Apriori Algorithm
    """

    def association_rules(self):
        """
        Returns the association rules.
        :return: List of Tuples (item_set, association, confidence)
        """
        return self.produce_association_rules()

    @abc.abstractmethod
    def find_n_frequent_item_sets(self):
        """
        Fines the N-Frequent items-sets in the transaction matrix.
        """
        pass

    @abc.abstractmethod
    def produce_association_rules(self):
        """
        Finds the associations rules whose confidences >= min_confidence.
        Associations are stores in a list attribute.
        """
        pass

    @abc.abstractmethod
    def has_infrequent_subset(self, candidate_item_set):
        """
        Finds if the candidate item-set contains and infrequent subset.
        :param candidate_item_set: Item-set that can be frequent.
        :return: True if it contains an infrequent subset, then
        candidate_item_set is infrequent too. False if doesn't have an
        infrequent subset.
        """
        pass


class PandasAprioriAlgorithm(AprioriAlgorithm, ABC):
    """
    Defines Apriori Algorithm using Pandas DataFrames.
    """

    def __init__(self, data_frame, items_list, min_support=2, min_confidence=0.7):
        """
        Creates a class instance.
        :param data_frame: The transaction matrix as DataFrame
        :param items_list: The list of all items in the dataset.
        :param min_support: An integer value to set minimum support.
        :param min_confidence: A float values to set minimum confidence threshold.
        """
        self.data_frame = data_frame            # Transaction matrix
        self.items_list = items_list            # All items in the dataset
        self.min_support = min_support          # Item-set minimum support.
        self.min_confidence = min_confidence    # Item-set minimum confidence.
        self.frequent_item_set = []             # List of Tuples (item-set, frequency).
        self.infrequent_item_set = []           # Infrequent item-sets list.
        self.longest_frequent_item_set = []     # List of Tuples (item-set, frequency).
        self.association_rules_list = []

        # Process the item_set of each transaction.
        Utilities.add_transaction_item_sets(self.data_frame, self.items_list)

        self.item_sets = Utilities.get_items_sets(self.data_frame)
        self.consumed_items = self.consumed_items_series()
        self.consumed_items_list = list(self.consumed_items.index)

    def add_item_sets(self):
        """
        Creates the Item_sets column, if it was missed.
        """
        if Constants.ITEM_SETS in list(self.data_frame.columns):
            Utilities.add_transaction_item_sets(self.data_frame, self.items_list)

    def consumed_items_series(self):
        """
        Only the items that were consumed are useful for the algorithm.
        :return: A Series with non-zero values.
        """
        count_items = Utilities.sum_aggregation(self.data_frame, self.items_list)
        return count_items[count_items != 0]

    def run(self):
        """
        Runs the algorithm.
        """
        # Execute Apriori algorithm.
        self.find_n_frequent_item_sets()
        # Find the association rules.
        self.produce_association_rules()

    def find_n_frequent_item_sets(self):
        """
        Finds the n-frequent item-sets.
        It automatically stops ones there aren't more items-sets whose support >= min_support.
        """
        iteration = 1

        while True:
            # Calculate combinations of size n
            item_set_combinations = Utilities.create_combinations(self.consumed_items_list, n=iteration)
            # Prune the combinations: All subsets of a frequent item-set must also be frequent.
            potential_freq_item_set = self.candidate_item_sets_list(item_set_combinations)
            # Obtain frequency of each item-set.
            item_sets_frequency_df = self.get_item_sets_frequency(potential_freq_item_set)

            # Find the item sets whose frequency >= min_support.
            # This process also updates the frequent_item_set and infrequent_item_set lists.
            # If no more frequent_item_set are found, then break the loop.
            if self.find_frequent_item_sets(item_sets_frequency_df, min_support=self.min_support):
                iteration += 1
            else:
                break

    def association_rules(self):
        """
        Returns the found association rules.
        :return:
        """
        return self.association_rules_list

    def produce_association_rules(self):
        """
        Finds the associations rules whose confidences >= min_confidence.
        Associations are stores in a list attribute.
        """
        for item_set, support in self.longest_frequent_item_set:
            subsets = Utilities.create_subsets(item_set)

            for subset in subsets:
                diff = item_set.difference(subset)
                diff_support = self.find_item_set_support(diff)
                confidence = support / diff_support

                if confidence >= self.min_confidence:
                    self.association_rules_list.append((subset, diff, confidence))

    def find_item_set_support(self, item_set):
        """
        Find the item-set support into the frequent_item_set.
        :param item_set: The item-set to be searched.
        :return: Integer value that represents the item-set support.
        """
        for index, tup in enumerate(self.frequent_item_set):
            if tup[0] == item_set:
                return tup[1]

    def has_infrequent_subset(self, candidate_item_set):
        """
        Finds if the candidate item-set contains and infrequent subset.
        :param candidate_item_set: Item-set that can be frequent.
        :return: True if it contains an infrequent subset, then
        candidate_item_set is infrequent too. False if doesn't have an
        infrequent subset.
        """
        is_candidate = True

        for sub_set in self.infrequent_item_set:

            # If it's a subset of a non-frequent item set.
            # Then, it isn't a frequent item-set.
            if sub_set.issubset(candidate_item_set):
                is_candidate = False
                break

        return is_candidate

    def update_infrequent_list(self, all_item_sets, frequent_item_sets):
        """
        Updates the infrequent_item_set given the list of all items and frequent items.

        :param all_item_sets: All items-sets list.
        :param frequent_item_sets: Frequent item-sets list.
        """
        for item_set in all_item_sets:
            if item_set not in frequent_item_sets:
                self.infrequent_item_set.append(item_set)

    def find_frequent_item_sets(self, items_df, min_support=2):
        """
        Finds those items_sets whose >= min_support given a 2-columns DataFrame
        which contains the frequency of each item-set.
        It also updates the frequent_item_set, infrequent_item_set and frequent_item_set_count lists.

        :param items_df: 2-columns DataFrame: Item_sets, Count
        :param min_support: Integer value.
        :return: True if any item_set frequency >= min_support.
        """
        # Filter items whose frequency >= min_support.
        freq_items_series = items_df[items_df[Constants.COUNT] >= min_support]

        if freq_items_series.shape[0] > 0:
            # Update infrequent items.
            # They are used to prune bigger item-sets
            self.update_infrequent_list(
                list(items_df[Constants.ITEM_SETS]),
                list(freq_items_series[Constants.ITEM_SETS]))

            # Update frequent items.
            self.longest_frequent_item_set = list(freq_items_series.apply(tuple, axis=1))
            self.frequent_item_set += self.longest_frequent_item_set

            return True
        else:
            return False

    def candidate_item_sets_list(self, candidate_item_sets):
        """
        Filter out new sub sets.
        Based on the Apriori property that all subsets of a frequent item-set must also be frequent

        :param candidate_item_sets: The new item sets to filter out.
        :return: item-sets that can be frequent.
        """
        potential_result = []
        new_non_freq = []

        for candidate_item_set in candidate_item_sets:
            if self.has_infrequent_subset(candidate_item_set):
                potential_result.append(candidate_item_set)
            else:
                new_non_freq.append(candidate_item_set)

        # Update infrequent_item_set.
        # They are used to prune bigger item-sets
        self.infrequent_item_set += new_non_freq

        return potential_result

    def get_item_sets_frequency(self, items_sets_list):
        """
        Finds the frequency of appearance of each candidate item-set in the transaction DataFrame.

        :param items_sets_list: List of item_sets to be counted.
        :return: A 2-columns DataFrame: Item_sets, Count.
        """

        count_item_sets = {}
        item_sets = {}

        for item_set in items_sets_list:
            for user_item_set in self.item_sets:
                # Count each time item-set appears.
                if item_set.issubset(user_item_set):
                    key = str(item_set)
                    item_sets[key] = item_set
                    if key in count_item_sets:
                        count_item_sets[key] = count_item_sets[key] + 1
                    else:
                        count_item_sets[key] = 1

        # Create a 2-columns DataFrame: Item_sets, Count
        # DataFrame are easy to manage. However, this can be changed.
        return self.convert_to_data_frame(count_item_sets, item_sets)

    def frequent_item_set_as_df(self):
        """
        Returns frequent item-sets as DataFrame.
        The columns are item_set and support
        :return: DataFrame.
        """
        return pd.DataFrame(
            self.frequent_item_set,
            columns=[Constants.ITEM_SETS, Constants.SUPPORT])

    def associations_as_df(self):
        """
        Returns the Association as DataFrame.
        The columns are item_set, association and confidence
        :return: DataFrame
        """
        columns = [Constants.CONSUMED_ITEMS, Constants.FREQUENT_ITEMS, Constants.CONFIDENCE]
        return pd.DataFrame(self.association_rules_list, columns=columns).sort_values(
            by=[Constants.CONFIDENCE],
            ascending=False)

    @staticmethod
    def convert_to_data_frame(count_item_sets, item_sets):
        """
        Given two same size dict with same keys. It creates a 2-columns DataFrame.

        :param count_item_sets: The dict containing str(item_set): frequency
        :param item_sets: The dict containing str(item_set): {item_set}
        :return: A 2-columns DataFrame: Item_sets, Count.
        """
        col1 = []
        col2 = []

        for key in count_item_sets:
            col1.append(item_sets[key])
            col2.append(count_item_sets[key])

        # First DataFrame column are item-sets, second column are their frequency.
        data = {Constants.ITEM_SETS: col1, Constants.COUNT: col2}

        return pd.DataFrame(data=data).sort_values(by=[Constants.COUNT], ascending=False)
