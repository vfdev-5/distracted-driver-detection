
#
# Script to compare submissions
#

# Python
import os

# Numpy
import numpy as np

# pandas
import pandas as pd

###########################################################################
# DEFINE GLOBAL TRAINING PARAMETERS
###########################################################################

master_submission = "results/submission_2016-06-09-14-11__10_1750_150_8.csv"

# test identical
# slave_submission = "results/submission_2016-06-09-14-11__10_1750_150_8.csv"
# No predictions by master/slave 54476 54476

slave_submission = "results/submission_2016-06-09-09-40__10_-10_100_8.csv"
# Number of identical predictions 35092
# New prediction from slave submission 0
# Lost prediction by slave submission 9061
# Prediction precision improvement by slave submission 0
# Prediction precision decrease by slave submission 1496
# Bad prediction by slave submission 0
# No predictions by master/slave 54476 41771

# slave_submission = "results/submission_2016-06-09-14-40__10_2500_200_8.csv"

# slave_submission = "results/submission_2016-06-08-18-01__10_1500_100_8.csv"

threshold = 0.6
identity_threshold = 0.1

###########################################################################
# Check for existing files
###########################################################################

assert os.path.exists(master_submission), "Master submission file is not found"
assert os.path.exists(slave_submission), "Slave submission file is not found"


###########################################################################
# Load files as pandas DataFrames
###########################################################################

master_df = pd.read_csv(master_submission, ",")
slave_df = pd.read_csv(slave_submission, ",")

assert master_df.shape == slave_df.shape, "Data shape are different"
print "Data shape : ", master_df.shape

# Sort by image name
master_df = master_df.sort_values(by='img')
slave_df = slave_df.sort_values(by='img')

# print "master DF"
# print master_df.head()
#
# print "slave DF"
# print slave_df.head()


# Apply a threshold to drop incertain decisions
thresh = lambda x: x if x > threshold else 0.0
proc_m_df = master_df.iloc[:, 1:].applymap(thresh)
proc_s_df = slave_df.iloc[:, 1:].applymap(thresh)

n_no_predictions_master = proc_m_df[proc_m_df.sum(axis=1) == 0].shape[0]
n_no_predictions_slave = proc_s_df[proc_s_df.sum(axis=1) == 0].shape[0]

diff_df = proc_m_df - proc_s_df
# Remove all zero rows
diff_df = diff_df[diff_df.sum(axis=1) != 0]

if diff_df.empty:
    print "Submissions are identical"
    print 'No predictions by master/slave', n_no_predictions_master, n_no_predictions_slave
    exit(1)

# print "Diff DF"
# print diff_df.head()

score = lambda x: int(x*10) if x > identity_threshold else 0
score_df = diff_df.applymap(score)
# print "Score DF"
# print score_df.head(n=20)

t = int(threshold*10)

# 8 cases :
# 1) New prediction from slave submission <=> negative value <= -t
# 2) Lost prediction by slave submission <=> positive value >= t

# 3) Prediction precision improvement <=>  negative value > -t
# 4) Prediction precision decrease <=>  positive value < t

# 5) Bad prediction by slave submission <=> two values : one positive and one negative

# 6) No predictions by master
# 7) No predictions by slave
# 8) Number of identical

n_new_predictions = 0
n_lost_predictions = 0
n_predictions_improuv = 0
n_predictions_decrease = 0
n_bad_predictions = 0
n_same_predictions = 0

# not efficient solution
print "Compute stats on {} rows ...".format(score_df.shape[0])
for index, row in score_df.iterrows():
    row = row.values
    s = np.sum(row)
    count = len(row[row != 0])
    if count == 2:
        n_bad_predictions += 1
        continue
    if s == 0:
        n_same_predictions += 1
    elif s <= -t:
        n_new_predictions += 1
    elif -t < s < 0:
        n_predictions_improuv += 1
    elif s >= t:
        n_lost_predictions += 1
    elif 0 < s < t:
        n_predictions_decrease += 1

print 'Number of identical predictions', n_same_predictions
print 'New prediction from slave submission', n_new_predictions
print 'Lost prediction by slave submission', n_lost_predictions
print 'Prediction precision improvement by slave submission', n_predictions_improuv
print 'Prediction precision decrease by slave submission', n_predictions_decrease
print 'Bad prediction by slave submission', n_bad_predictions
print 'No predictions by master/slave', n_no_predictions_master, n_no_predictions_slave

print "Check : ", slave_df.shape[0], n_same_predictions + n_new_predictions + n_predictions_improuv + n_predictions_decrease + n_bad_predictions + n_no_predictions_slave
