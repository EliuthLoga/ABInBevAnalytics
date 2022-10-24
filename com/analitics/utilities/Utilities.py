import numpy as np
import pandas as pd

from datetime import datetime
from itertools import combinations

from com.analitics.utilities import Constants


"""
This module contains useful methods.
"""


def read_parquet(file_name, engine="pyarrow"):
    """
    Reads parquet file.
    :param file_name: The parquet file name to be read.
    :param engine: The engine used. Default = pyarrow.
    :return: Pandas DataFrame
    """

    return pd.read_parquet(file_name, engine=engine)


def print_df_column_nunique(data_frame):
    """
    Prints number of unique elements for each DataFrame column.

    Excludes NA values by default.
    :param data_frame: Pandas DataFrame.
    """
    for column_name in data_frame:
        print(column_name + ": " + str(data_frame[column_name].nunique()))


def print_df_column_unique(data_frame):
    """
    Prints unique elements for each DataFrame column.

    :param data_frame: Pandas DataFrame.
    """
    for column_name in data_frame:
        print(data_frame[column_name].unique())


def describe(data_frame, column_name, as_categorical=False):
    """
    Generate descriptive statistics of the given column name.

    :param column_name: The column name to be described.
    :param as_categorical: If column data type is numerical, but they must be treated as categorical.
    :return: A DataFrame
    """
    # Treat the values as categorical.
    if as_categorical:
        dict_data = data_frame[column_name].value_counts().to_dict()

        data = {
            column_name: [
                data_frame.shape[0],
                data_frame[column_name].nunique(),
                list(dict_data.keys())[0],
                list(dict_data.values())[0]]}

        # Create a similar DataFrame as DataFrame.describe()
        return pd.DataFrame(data=data, index=["count", "unique", "top", "freq"])
    else:
        return data_frame[column_name].describe()


def any_null_value(data_frame):
    """
    Finds if the DataFrame contains any null value.
    :return: True if DataFrame contains any null value.
    """
    return data_frame.isnull().values.any()


def has_null_value(data_frame):
    """
    Searches for null values in the DataFrame.
    :return: True if DataFrame contains any null value.
    """
    result = []
    for column_name in data_frame:
        if data_frame[column_name].isnull().values.any():
            result.append(column_name)

    return result


def aggregate_data_frame(data_frame, agg_functions, group_by_col):
    """
    Performs aggregation in the given DataFrame.
    :param data_frame: The DataFrame.
    :param agg_functions: The aggregation function.
    :param group_by_col: The columns to perform group by.
    :return:
    """
    return data_frame.groupby(group_by_col).aggregate(agg_functions)


def convert_to_datetime(string_date, date_format="%Y-%m-%d"):
    """
    This function converts to string to date time.
    :param string_date: String to be converted.
    :param date_format: Date format.
    :return:
    """
    return datetime.strptime(string_date, date_format)


def add_date_information(data_frame, date_column, date_format="%Y-%m-%d"):
    """
    Adds ddate information to a given DataFrame. Dataframe must contain
    the string values in a column.
    :param data_frame: DataFrame
    :param date_column: Date column
    :param date_format: Date format.
    """
    data_frame[Constants.ORDER_DATE] = pd.to_datetime(
        data_frame[date_column],
        format=date_format)

    data_frame[Constants.DAY_OF_WEEK] = data_frame[Constants.ORDER_DATE].dt.day_of_week
    data_frame[Constants.DAY_OF_MONTH] = data_frame[Constants.ORDER_DATE].dt.day
    data_frame[Constants.MONTH] = data_frame[Constants.ORDER_DATE].dt.month


def index_of(np_array, value):
    """
    Gets the index of the given value in the numpy array.
    :param np_array: Numpy array.
    :param value: Value to be searched.
    :return: Index of the item.
    """
    return np.where(np_array == value)[0][0]


def aggregate_df(data_frame, group_by_list, aggregation_col):
    """
    This method aggregates the columns that are only different by Quantity column value.
    :param data_frame: The DataFrame.
    :param group_by_list:
    :param aggregation_col:
    :return: DataFrame
    """
    column_names = list(data_frame.columns.values)
    unprocessed_cols = group_by_list.copy() + [aggregation_col]
    skip_columns = set(column_names).difference(set(unprocessed_cols))
    orders_df = data_frame.groupby(group_by_list)

    data_frame_data = []
    for name, unused_df in orders_df:
        row = list(name)

        # Pass the first value of the skip columns
        for skip_column in skip_columns:
            row.append(unused_df.iloc[0][skip_column])

        # Aggregate using SUM.
        row.append(unused_df[aggregation_col].sum())

        # Store row information.
        data_frame_data.append(row)

    return pd.DataFrame(data=data_frame_data, columns=column_names)


def create_transaction_df(data_frame, group_by_list, items_col, quantity_col):
    """
    This method aggregates the columns that are only different by Quantity column value.
    :param data_frame: The DataFrame.
    :param group_by_list: The columns to perform group by.
    :param items_col: The items column
    :param quantity_col: The Quantity Column
    :return: DataFrame
    """
    items = np.sort(data_frame[items_col].unique())

    # Group by date and account id, or any useful attribute
    orders_df = data_frame.groupby(group_by_list)
    data_frame_data = []

    for name, unused_df in orders_df:

        # Get quantity and item_id column, sort it by item_ids.
        # Sorting can help to find frequent items easily.
        sorted_data = unused_df[[items_col, quantity_col]].sort_values(
            by=[items_col],
            axis=0,
            ascending=True)

        # Start the process to create the list of items per order.
        sordered_items = set(sorted_data[items_col])
        transaction_items = [0] * items.size

        for product in sordered_items:
            transaction_items[index_of(items, product)] = 1

        # Store useful information such as total quantity, quantity-items.,
        row = list(name) + transaction_items
        row.append(sorted_data[quantity_col].sum())

        # Store order information
        data_frame_data.append(row)

    items_column_names = [str(item_id) for item_id in items]

    return pd.DataFrame(
        data=data_frame_data,
        columns=group_by_list + items_column_names + [Constants.TOTAL_QUANTITY])


def create_full_transaction_df(data_frame, group_by_list, items_col, quantity_col):
    """
    This method aggregates the columns that are only different by Quantity column value.
    :param data_frame: The DataFrame.
    :param group_by_list: The columns to perform group by.
    :param items_col: The items column
    :param quantity_col: The Quantity Column
    :return: DataFrame
    """
    items = np.sort(data_frame[items_col].unique())

    # Group by date and account id, or any useful attribute
    orders_df = data_frame.groupby(group_by_list)
    data_frame_data = []

    for name, unused_df in orders_df:

        # Get quantity and item_id column, sort it by item_ids.
        # Sorting can help to find frequent items easily.
        sorted_data = unused_df[[items_col, quantity_col]].sort_values(
            by=[items_col],
            axis=0,
            ascending=True)

        # Start the process to create the list of items per order.
        sordered_items = set(sorted_data[items_col])
        order_items = [0] * items.size

        for product in sordered_items:
            order_items[index_of(items, product)] = 1

        # Store useful information such as total quantity, quantity-items.,
        row = list(name) + order_items
        row.append(sordered_items)
        row.append(len(sordered_items))
        row.append(list(sorted_data[quantity_col]))
        row.append(sorted_data[quantity_col].sum())

        # Store order information
        data_frame_data.append(row)

    items_column_names = [str(item_id) for item_id in items]

    return pd.DataFrame(
        data=data_frame_data,
        columns=
        group_by_list
        + items_column_names
        + [Constants.ITEM_SETS, Constants.NUMBER_ITEMS, Constants.QUANTITY_SETS, Constants.TOTAL_QUANTITY])


def add_transaction_item_sets(transaction_df, items_ids):
    """
    Calculates the item_set column for the given transaction  matrix.
    :param transaction_df: The transaction matrix.
    :param items_ids: the items ids,
    """

    item_sets = []
    transaction_items = transaction_df[items_ids]

    for index in transaction_items.index:
        series = transaction_items.loc[index]
        series = series[series != 0]
        item_sets.append(set(series.index))

    transaction_df[Constants.ITEM_SETS] = item_sets


def create_subsets(item_set):
    """
    Create subsets of the given item.
    :param item_set: The item used to generate subsets.
    :return: The list of subsets.
    """

    subsets = []
    for n in range(1, len(item_set)):
        subsets += create_combinations(item_set, n)

    return subsets


def create_combinations(iterable, n):
    """
    Create combinations of size n, using iterable item.
    :param iterable: The iterable item.
    :param n: The length of the generated item-sets.
    :return: Combinations of size n.
    """
    result = []

    for c in combinations(iterable, n):
        result.append(set(sorted(list(c))))

    return result


def get_user_history_df(data_frame, user_id, filter_date):
    """
    Filter the given DataFrame based on user_ud and date.
    :param data_frame: The DataFrame to be filtered.
    :param user_id: The user id to recommend.
    :param filter_date: The filter date.
    :return:
    """
    return data_frame[(data_frame[Constants.ACCOUNT_ID] == user_id) &
                      (data_frame[Constants.DATE] < filter_date)]


def get_items_sets(data_frame):
    """
    Returns the list of items values.
    :param data_frame: The data_frameñ.
    :return: List of item ids.
    """

    return list(data_frame[Constants.ITEM_SETS].values)


def series_to_dict(series):
    """
    Converts a given series in a dictionary.
    :param series: The Series to be converted.
    :return: A dictionary.
    """
    result_dict = {}

    for item_id in series.index:
        result_dict[str({item_id})] = int(series[item_id])

    return result_dict


def sum_aggregation(data_frame, all_items_columns, axis=0):
    """
    Performs sum aggregation to the given DataFrame.
    :param data_frame: DataFrame
    :param all_items_columns: ALll items columns to retrieve transaction matrix.
    :param axis: The axis to perform aggregation.
    :return: A DataFrame.
    """
    return data_frame[all_items_columns].sum(axis=axis).sort_values(ascending=False)


def date_filter_dataframe(data_frame, limit_date):
    """
    Filter DataFrame by date.
    :param data_frame: DataFrame to be filtered.
    :param limit_date: The limit day.
    :return: A filtered DataFrame.
    """
    return data_frame[data_frame[Constants.DATE] < limit_date]


def product_group_by(data_frame, group_by):
    """
    Group by product id DataFrame and then group by column
    :param data_frame: Data_frame used.
    :param group_by: a second criteria for group by-
    :return: Filtered DataFrame.
    """

    return data_frame[[Constants.PRODUCT_ID, group_by]].groupby([Constants.PRODUCT_ID])