from com.analitics.utilities import Constants


class ParquetCleaning:
    """
    Defines a class with the steps to clean the dataset.
    """
    def __init__(self, parquet_data):
        self.parquet_data = parquet_data

    def clean(self, root, limit_value):
        """
        Cleans the data loaded from parquet.
        :param root: Root log.
        :param limit_value: The limit value.
        """
        root.info("Removing rows with Quantity = 0.")
        self.parquet_data.remove_rows_zero_val(Constants.QUANTITY)

        root.info("Removing duplicated rows.")
        self.parquet_data.remove_duplicates()

        root.info("Removing rows whose Quantity >= limit_value.")
        self.parquet_data.filter_rows_by_quantity(Constants.QUANTITY, limit_value)



