import numpy as np
import implicit
import music_fairness_lib as mflib
from scipy import sparse

# Load data
a_u_tuple = mflib.user_events_file_to_lil_matrix()
a_u_matrix = a_u_tuple[0]

# parameters to try
alpha_list = [4, 16, 32, 64]
reg_list = [.01, .1, 1, 10]
factor_list = [64, 128, 256]

# store outcomes
out_file = open("als_hyperparameters.txt", "a+")
out_file.write("alpha\treg\tfactors\trec_auc\tpop_auc\n")

# train test split
u_to_a_train, u_to_a_test, altered_users = mflib.make_train(a_u_matrix.T.tocsr(), pct_test=0.2)

# iterate through every possible hyper-parameter combination
for alpha_idx in range(len(alpha_list)):
    for reg_idx in range(len(reg_list)):
        for factor_idx in range(len(factor_list)):
            print(alpha_idx, reg_idx, factor_idx)

            # split original matrix into user matrix and artist matrix through ALS
            user_vecs, artists_vecs = implicit.alternating_least_squares(
                (u_to_a_train * alpha_list[alpha_idx]).astype('double'),
                factors=factor_list[factor_idx],
                regularization=reg_list[reg_idx],
                iterations=50,
                use_gpu=True)

            # calculate mean ROC area under curve
            rec_auc, pop_auc = mflib.calc_mean_auc(u_to_a_train, altered_users,
                                             [sparse.csr_matrix(user_vecs), sparse.csr_matrix(artists_vecs.T)],
                                             u_to_a_test)

            # write to file
            out_file.write(str(alpha_list[alpha_idx]) + "\t" + str(reg_list[reg_idx]) + "\t" + str(
                factor_list[factor_idx]) + "\t" + str(rec_auc) + "\t" + str(pop_auc) + "\n")

out_file.close()

