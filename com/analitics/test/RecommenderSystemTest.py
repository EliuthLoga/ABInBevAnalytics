import logging
import sys

import numpy as np
import pandas as pd

from com.analitics.algorithms.FrequentItemSets import PandasAprioriAlgorithm
from com.analitics.application.RecommendationApp import get_cleaned_dataset, get_transaction_dataset
from com.analitics.recommendation.RecommendationEngines import UserFrequentItemsEngine
from com.analitics.utilities import Utilities, Constants
from com.analitics.utilities.ClassificationAccuracy import precision, recall


def precision_recall(real_items, predicted_items):
    """
    Calclates precision and recall of the predictions.
    :param real_items: The real items DataFrame
    :param predicted_items: The predictions DataFrame
    :return: Tuple: (precision, recall)
    """
    true_positives = real_items.intersection(predicted_items)
    false_negatives = real_items.difference(predicted_items)
    false_positive = predicted_items.difference(true_positives)

    return (precision(true_positives, false_positive), recall(true_positives, false_negatives))


def prediction_evaluation(real_transactions, recommendations):
    """
    Evaluates the predictions. It uses precision, recall and MAPE.
    :param real_transactions: The real items DataFrame
    :param recommendations: The predictions DataFrame
    :return: Tuple: (precision, recall, MAPE)
    """
    real_items = set(real_transactions[Constants.PRODUCT_ID].values)
    real_items = set([str(item_id) for item_id in real_items])
    predicted_items = set(recommendations[Constants.PRODUCT_ID].values)

    precision_val, recall_val = precision_recall(real_items, predicted_items)

    error = []
    for product in real_transactions[Constants.PRODUCT_ID]:
        real_quantity = real_transactions[
            real_transactions[Constants.PRODUCT_ID] == product][Constants.QUANTITY].values[0]
        find_recommendation = recommendations[
            recommendations[Constants.PRODUCT_ID] == str(product)
            ][Constants.RECOMMENDED_QUANTITY]

        if find_recommendation.shape[0] > 0:
            predicted_quantity = find_recommendation.values[0]
        else:
            predicted_quantity = 0

        diff_perc = (abs(real_quantity - predicted_quantity) / abs(real_quantity)) * 100
        error.append(diff_perc)

    return (precision_val, recall_val, round(sum(error) / len(error), 3))


def generate_recommendations(
        user_id, transactions_history,
        history_data, items_ids,
        general_rec, n_recommendations=10,
        initial_top=3, min_support=3,
        min_confidence=0.7):
    """
    Generates recommendations for a user.
    :param user_id: User_id
    :param transactions_history: User transaction history.
    :param history_data: The user shopping history.
    :param items_ids: The total items_ids.
    :param general_rec: General recommendations.
    :param n_recommendations: The number of recommendations.
    :param initial_top: The initial top to exploit Association Rules.
    :param min_support: The minimum_support.
    :param min_confidence: The minimum_conffidence.
    :return: A list of recommendations.
    """
    user_recommendations = []
    user_recommended_items = []

    # If there is a single row, then produce personalized recommendation.
    # If not then consider all users preferences.
    if user_history_transactions.shape[0] > 1:
        # Use Apriori Algorithm
        algorithm = PandasAprioriAlgorithm(
            transactions_history,
            items_ids,
            min_support=min_support,
            min_confidence=min_confidence)

        engine = UserFrequentItemsEngine(
            user_id,
            history_data,
            transactions_history)
        engine.generate_recommendations(algorithm, k=initial_top)

        user_recommendations += engine.recommendations
        user_recommended_items += engine.recommended_items

    for recommendation in general_rec:
        if recommendation[0] not in user_recommended_items:
            user_recommendations.append((user_id,) + recommendation)
            user_recommended_items.append(recommendation)

    user_recommendations = user_recommendations[:n_recommendations]

    columns = [
        Constants.ACCOUNT_ID,
        Constants.PRODUCT_ID,
        Constants.RECOMMENDED_QUANTITY,
        Constants.DAY_OF_WEEK]

    return pd.DataFrame(user_recommendations, columns=columns)


if __name__ == "__main__":
    # Initialize Logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

    cleaned_data_file = "../data/CleanedData.csv"
    transactions_data_file = "../data/TransactionsData.csv"

    # Load cleaned data and transaction matrix.
    cleaned_dataset_df = get_cleaned_dataset(root, cleaned_data_file)
    transactions_df = get_transaction_dataset(root, transactions_data_file)

    items = list(np.sort(cleaned_dataset_df[Constants.PRODUCT_ID].unique()))
    items_string_ids = [str(item_id) for item_id in items]

    Utilities.add_date_information(transactions_df, Constants.DATE)
    Utilities.add_date_information(cleaned_dataset_df, Constants.DATE)

    n_users_to_recommend = 10  # Users to randomly take and make the predictions.
    recommendations_number = 10
    support = 3  # Frequent item-set support.
    confidence = 0.7  # Association rules confidence.
    top = 4  # Top K to recommend.
    random_users = False  # If use random_users.
    month_test = 7

    test_data = transactions_df[transactions_df[Constants.MONTH] == month_test]

    test_users = set([33237129, 33611229, 33236163, 33238638, 40620450, 33710592, 33240375, 37545453,
                  33259770, 40518873, 33256614, 33256614, 33247263, 34655979, 33899601, 33640824,
                  33262227, 33227430, 35226321, 33254319, 33236958, 33253422, 38232621, 38232621])

    # users = list(np.sort(test_data[Constants.ACCOUNT_ID].unique()))
    # k = int(len(users) * 0.2)
    # test_users = random.sample(users, 20)

    top_k_rec = Utilities.top_k_items_recommendations(
        cleaned_dataset_df,
        Constants.PRODUCT_ID,
        k=recommendations_number*2)         # Load 2 times the recommendation number

    total_evaluation = []
    print(f"Total Users: {len(test_users)}")
    for user_id in test_users:
        user_history_df = transactions_df[transactions_df[Constants.ACCOUNT_ID] == user_id]

        predicting_transactions = transactions_df[
            (transactions_df[Constants.ACCOUNT_ID] == user_id) &
            (transactions_df[Constants.MONTH] == month_test)]

        print(f"User ID: {user_id}, Transaction: {predicting_transactions.shape[0]}")

        user_evaluation = []

        for transaction_id in predicting_transactions.index:
            test_transaction_information = predicting_transactions.loc[transaction_id]
            transaction_date = test_transaction_information[Constants.DATE]
            transaction_week_day = test_transaction_information[Constants.DAY_OF_WEEK]

            # Get detailed information of the transaction.
            transaction_detailed_information = cleaned_dataset_df[
                (cleaned_dataset_df[Constants.ACCOUNT_ID] == user_id) &
                (cleaned_dataset_df[Constants.DATE] == transaction_date)][
                [Constants.PRODUCT_ID, Constants.QUANTITY, Constants.DATE, Constants.DAY_OF_WEEK]]

            transaction_items = transaction_detailed_information[Constants.PRODUCT_ID].values

            # User history information of the buys of the user before limit_date
            user_history_data = Utilities.get_user_history_df(
                cleaned_dataset_df,
                user_id,
                transaction_date).copy()

            # Transactions of the buys of the user before limit_date
            user_history_transactions = Utilities.get_user_history_df(
                transactions_df,
                user_id,
                transaction_date).copy()

            # Generate Recommendations
            recommendations = generate_recommendations(
                user_id,
                user_history_transactions,
                user_history_data,
                items_string_ids,
                top_k_rec,
                recommendations_number,
                initial_top=top,
                min_support=support,
                min_confidence=confidence)

            # Evaluate Predictions
            user_evaluation.append(
                prediction_evaluation(
                    transaction_detailed_information,
                    recommendations))

        average = pd.DataFrame(data=user_evaluation).mean(axis=0)
        total_evaluation.append(list(average.values))

    total_eval_df = pd.DataFrame(data=total_evaluation, columns=["Precision", "Recall", "MAPE"])
    print(total_eval_df.mean(axis=0))
