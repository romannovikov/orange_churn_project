def count_row_nans(data):
    return data.T.isna().sum()
