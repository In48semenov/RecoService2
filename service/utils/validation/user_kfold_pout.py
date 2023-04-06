import numpy as np


class UsersKFoldPOut:
    def __init__(
        self,
        n_folds: int,
        p: int,
        user_column: str = 'user',
        order_column: str = 'order',
        random_seed: int = 23
    ):
        self.n_folds = n_folds
        self.p = p
        self.user_column = user_column
        self.order_column = order_column
        self.random_seed = random_seed

    def split(self, df):
        users = df[self.user_column].unique()
        users_count = len(users)

        np.random.seed(self.random_seed)
        np.random.shuffle(users)

        fold_sizes = np.full(
            self.n_folds, users_count // self.n_folds, dtype=int
        )
        fold_sizes[:users_count % self.n_folds] += 1
        curr_idx = 0
        for fold_size in fold_sizes:
            start_idx, stop_idx = curr_idx, curr_idx + fold_size
            test_fold_users = users[start_idx:stop_idx]
            test_mask = df[self.user_column].isin(test_fold_users)
            train_mask = ~test_mask
            test_mask &= df.loc[test_mask, self.order_column] < self.p

            curr_idx += fold_size

            yield train_mask, test_mask
