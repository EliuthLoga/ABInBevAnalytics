from abc import abstractmethod

import pandas as pd


class ParquetData:
    """
    This class defines objects used to read parquet data.
    """
    @abstractmethod
    def read_parquet(self, file_name, engine):
        """
        Reads parquet file.

        :param file_name: The parquet file name to be read.
        :param engine: The engine used. Default = pyarrow.
        :return: Pandas DataFrame
        """
        pass

    @abstractmethod
    def describe(self, column_name, as_categorical):
        """
        Generate descriptive statistics of the given column name.

        :param column_name: The column name to be described.
        :param as_categorical: If column data type is numerical, but they must be treated as categorical.
        :return: A DataFrame
        """
        pass

    @abstractmethod
    def remove_duplicates(self):
        """
        Removes duplicates rows.
        """
        pass

    @abstractmethod
    def get_data_frame(self):
        """
        Return the associated Pandas DataFrame to this class.
        :return: Pandas DataFrame.
        """
        pass

    @abstractmethod
    def any_null_value(self):
        """
        Finds if the DataFrame contains any null value.
        :return: True if DataFrame contains any null value.
        """
        pass

    @abstractmethod
    def has_null_value(self):
        """
        Searches for null values in the DataFrame.
        :return: True if DataFrame contains any null value.
        """
        pass


class ParquetPandasDF(ParquetData):
    """
    Defines a Parquet reading class using Pandas DataFrame.
    """

    def __init__(self, parquet_file, engine="pyarrow"):
        """
        Constructor.
        :param parquet_file: The file path.
        :param engine: The engine to read parquet file. Default pyarrow.
        """
        self.data_frame = self.read_parquet(parquet_file, engine)

    def read_parquet(self, file_name, engine):
        """
        Reads parquet file.

        :param file_name: The parquet file name to be read.
        :param engine: The engine used. Default = pyarrow.
        :return: Pandas DataFrame
        """
        return pd.read_parquet(file_name, engine=engine)

    def describe(self, column_name, as_categorical=False):
        """
        Generate descriptive statistics of the given column name.

        :param column_name: The column name to be described.
        :param as_categorical: If column data type is numerical, but they must be treated as categorical.
        :return: A DataFrame
        """
        # Treat the values as categorical.
        if as_categorical:
            dict_data = self.data_frame[column_name].value_counts().to_dict()

            d = {column_name: [
                self.data_frame.shape[0],
                self.data_frame[column_name].nunique(),
                list(dict_data.keys())[0],
                list(dict_data.values())[0]]}
            return pd.DataFrame(data=d, index=["count", "unique", "top", "freq"])
        else:
            return self.data_frame[column_name].describe()

    def any_null_value(self):
        """
        Finds if the DataFrame contains any null value.
        :return: True if DataFrame contains any null value.
        """
        return self.data_frame.isnull().values.any()

    def has_null_value(self):
        """
        Searches for null values in the DataFrame.
        :return: True if DataFrame contains any null value.
        """
        result = []
        for column_name in self.data_frame:
            if self.data_frame[column_name].isnull().values.any():
                result.append(column_name)

        return result

    def remove_duplicates(self):
        """
        Inplace operation to remove duplicated rows.
        """
        self.data_frame.drop_duplicates(keep="first", inplace=True)

    def filter_rows_by_quantity(self, column_name, limit):
        """
        Removes rows whose quantity < than limit.
        """
        self.data_frame.drop(self.data_frame[(self.data_frame[column_name] < limit)].index, inplace=True)

    def remove_rows_zero_val(self, column_name):
        """
        Removes rows with zero value in the given column.
        """
        self.data_frame.drop(self.data_frame[(self.data_frame[column_name] == 0)].index, inplace=True)

    def get_data_frame(self):
        """
        Returns the DataFrame.
        :return: Pandas DataFrame
        """
        return self.data_frame
