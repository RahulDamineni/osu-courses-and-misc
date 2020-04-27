import numpy as np
import pathlib
import time

DATA_DIR = pathlib.Path("./hw1-data")

train_path = DATA_DIR / "income.train.txt.5k"
eval_path = DATA_DIR / "income.dev.txt"
test_path = DATA_DIR / "income.test.blind"
train_and_eval_path = DATA_DIR / "income.combined.6k"

COL_NAMES = [
    "age",
    "sector",
    "education",
    "marital-status",
    "occupation",
    "race",
    "gender",
    "hours-per-week",
    "country-of-origin",
    "target"
]


def txt_file_to_df(file_path):

    DELIMITER = ","

    def parse_row(row, delimiter=DELIMITER):
        cells = row.split(delimiter)
        parsed_row = [
            int(cell.strip())
            if cell.isnumeric()
            else cell.strip()
            for cell in cells
        ]

        return parsed_row

    data = []
    with open(file_path) as in_:
        raw_rows = in_.readlines()

    for row in raw_rows:
        parsed_row = parse_row(row)
        data.append(parsed_row)

    df = {col: val for col, val in zip(*[COL_NAMES, zip(*data)])}
    return df, data


def txt_to_encoded_df(file_path, encoders):

    # Load and parse data
    df, _ = txt_file_to_df(file_path)

    # Encode and transform each col
    encoded_df = []
    for col in df.keys():

        if col == "target":
            continue

        elif col in encoders:
            encoder = encoders[col]
            col_values = encoder.transform(df[col])

        else:
            col_values = np.array(df[col], dtype=np.float64).reshape(len(df[col]), -1) / 50.

        encoded_df.append(col_values)

    # Make a flat dataset from all cols
    encoded_df = np.hstack(encoded_df)

    if "test" in str(file_path):
        return encoded_df

    return encoded_df, df["target"]


def make_encoders(categorical_columns, train_df):
    '''
    Takes categorical column names and df to make encoder objects using data
    '''

    catg_columns = categorical_columns.keys()

    # Init encoder objects
    col_encoders = {col: OneHotEncoder() for col in catg_columns}

    # Train encoders
    for col in catg_columns:
        col_values = col_encoders[col].fit(train_df[col])

    return col_encoders


class OneHotEncoder(object):

    def __init__(self):
        pass

    def fit(self, column):
        self.unique_categories = list(set(column))
        self.catg_index = {catg: cid for cid, catg in enumerate(self.unique_categories)}

    def transform(self, column):
        ct, ck = 0, []
        self.ohe_column = np.zeros((len(column), len(self.unique_categories)), dtype=np.float64)
        for i, catg in enumerate(column):
            try:
                j = self.catg_index[catg]
                self.ohe_column[i, j] = 1
            except KeyError:
                ct += 1
                ck.append(catg)

#         print(f'Failed {ct} times because of {ck}')
        return self.ohe_column

    def fit_transform(self, column):
        self.fit(column=column)

        return self.transform(column=column)


class KNN(object):

    def __init__(self, k, dist="eu"):
        self.k = k
        self.dist_type = "eu"

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.y_bin = self.binarize_y(y)

    @staticmethod
    def binarize_y(y):
        y_bin = y.copy()
        y_bin[y_bin == "<=50K"] = 0
        y_bin[y_bin != "0"] = 1
        y_bin = y_bin.astype(np.int64)

        return y_bin

    def predit(self, X):

        diff = self.X[:, np.newaxis] - X

        # Finding distances b/w vector pairs
        if self.dist_type == "eu":
            distances = np.linalg.norm(diff, axis=2)
        else:
            distances = np.sum(np.abs(diff), axis=2)

        # picking top k vectors for each pair in X
        if self.k == 1:
            top_indices = np.argmin(distances, axis=0)
            preds = self.y_bin[top_indices]
            return preds

        topk_indices = np.argpartition(distances, range(self.k), axis=0)[:self.k, :]
        topk_preds = self.y_bin[topk_indices[:, np.newaxis]]

        # Picking most popular vote among topk_preds
        counts = np.apply_along_axis(np.bincount, 0, np.squeeze(topk_preds, axis=1))
        preds = np.argmax(counts, axis=0)

        return preds

    def evaluate(self, X, y):
        pr = self.predit(X)

        y_bin = self.binarize_y(y)
        error_rate = np.mean(pr != y_bin)
        positive_rate = np.mean(pr == 1)

        return error_rate, positive_rate


if __name__ == "__main__":

    cols_catg = {
        "sector": 7,
        "education": 16,
        "marital-status": 4,
        "occupation": 14,
        "race": 5,
        "gender": 2,
        "country-of-origin": 39,
        "age": "*",
        "hours-per-week": "*"
    }

    combined_df_raw, _ = txt_file_to_df(train_and_eval_path)
    one_hot_encoders = make_encoders(categorical_columns=cols_catg, train_df=combined_df_raw)

    train_X, train_y = txt_to_encoded_df(file_path=train_path, encoders=one_hot_encoders)
    eval_X, eval_y = txt_to_encoded_df(file_path=eval_path, encoders=one_hot_encoders)
    test_X = txt_to_encoded_df(file_path=test_path, encoders=one_hot_encoders)

    print(train_X.shape, eval_X.shape, test_X.shape)

    for K in [3, 5, 7, 9, 99, 999, 5000]:
        start_time = time.time()
        knn = KNN(k=K)
        knn.fit(X=train_X, y=np.array(train_y))
        err, por = knn.evaluate(X=eval_X, y=np.array(eval_y))
        tr_err, tr_por = knn.evaluate(X=train_X, y=np.array(train_y))
        print(f'k={K}\t train_err: {tr_err*100:.2f}% (+:{tr_por*100:.2f}%) \
        dev_err: {err*100:.2f}% (+:{por*100:.2f}%)')
        # print(f'k={K}\t dev_err: {err:.2f}\t +: {por:.2f}\t elapsed: {time.time() - start_time:.2f}s.')
