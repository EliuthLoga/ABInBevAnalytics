import logging
import sys

import pandas as pd

from os.path import exists
from com.analitics.data.ParquetData import ParquetPandasDF
from com.analitics.process.DataClean import ParquetCleaning
from com.analitics.recommendation.RecommenderSystem import FrequentItemSetRecommenderSystem
from com.analitics.utilities import Constants, Utilities


def get_cleaned_dataset(root, cleaned_data_file):
    """
    Returns the cleaned dataset. If it was saved then it reads the csv file,
    If not, then it creates it and saves it.
    :param root: The log class.
    :param cleaned_data_file: The file name.
    :return: DataFrame
    """

    if exists(cleaned_data_file):
        root.info("Loading cleaned data CSV file.")
        cleaned_dataset_df = pd.read_csv(cleaned_data_file)
    else:
        root.info("Loading Parquet file.")
        parquet_pandas = ParquetPandasDF("../data/dataset_test.parquet")

        root.info(f"DataFrame size {parquet_pandas.data_frame.shape}.")
        root.info(f"DataFrame head:\n{parquet_pandas.data_frame.head()}")

        # Clean DataFraem
        ParquetCleaning(parquet_pandas).clean(root, 7000)

        root.info(f"DataFrame size {parquet_pandas.data_frame.shape}.")

        root.info("Aggregating rows by Date and Account_id.")
        cleaned_dataset_df = Utilities.aggregate_df(
            parquet_pandas.data_frame,
            [Constants.DATE, Constants.ACCOUNT_ID, Constants.PRODUCT_ID],
            Constants.QUANTITY)

        cleaned_dataset_df.to_csv(cleaned_data_file, index=False)

    Utilities.add_date_information(cleaned_dataset_df, Constants.DATE)

    root.info(f"Cleaned DataFrame size {cleaned_dataset_df.shape}.")

    return cleaned_dataset_df


def get_transaction_dataset(root, transaction_data_file):
    """
    Returns the transaction dataset. If it was saved then it reads the csv file,
    If not, then it creates it and saves it.
    :param root: The log class.
    :param transaction_data_file: The file name.
    :return: DataFrame
    """
    if exists(transaction_data_file):
        root.info("Loading transactions CSV file.")
        transactions_df = pd.read_csv(transaction_data_file)
    else:
        root.info("Creating orders DataFrame.")
        transactions_df = Utilities.create_transaction_df(
            cleaned_dataset_df,
            [Constants.DATE, Constants.ACCOUNT_ID],
            Constants.PRODUCT_ID,
            Constants.QUANTITY)

        transactions_df.to_csv(transactions_data_file, index=False)

    Utilities.add_date_information(transactions_df, Constants.DATE)

    root.info(f"Transactions DataFrame size {transactions_df.shape}.")

    return transactions_df


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

    # Recommender System Parameters
    users_to_recommend= 10      # Users to randomly take and make the predictions.
    recommendations_number = 10
    support = 3                     # Frequent item-set support.
    confidence = 0.7                # Association rules confidence.
    top = 4                         # Top K to recommend.
    random_users = False            # If use random_users.
    limit_date = "2022-07-01"       # Time threshold.

    top_k_rec = Utilities.top_k_items_recommendations(
        cleaned_dataset_df,
        Constants.PRODUCT_ID,
        k=recommendations_number * 2)

    recommender_system = FrequentItemSetRecommenderSystem(
        cleaned_dataset_df,
        transactions_df,
        limit_date,
        top_k_rec,
        n_users=users_to_recommend,
        n_recommendations=recommendations_number,
        top=top,
        support=support,
        confidence=confidence,
        random_users=random_users
    )

    columns = [
        Constants.ACCOUNT_ID,
        Constants.PRODUCT_ID,
        Constants.RECOMMENDED_QUANTITY,
        Constants.DAY_OF_WEEK]

    recommendations_df = pd.DataFrame(recommender_system.recommend_items(), columns=columns)
    recommendations_df.to_csv("../data/UsersRecommendations.csv", index=False)
    print(recommendations_df)
