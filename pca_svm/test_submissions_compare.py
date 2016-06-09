
#
# Unit test of submissions compare script
#

# Python
import os
import tempfile
import unittest

# Numpy
import numpy as np

# Pandas
import pandas as pd

# Project
from submissions_compare import main

#####################################################################################
#
# Possible outcomes for a row
# [1] no prediction => all proba < threshold
# [2] one incertain good prediction <=> 1.0 > proba > threshold
# [3] one bad incertain prediction <=> 1.0 > proba > threshold

# 9 possible combinations when compare two rows
# [1] ? [1] ==> n_no_predictions_master++, n_no_predictions_slave++
# [1] ? [2],[3] ==> n_no_predictions_master++, n_new_predictions++ by slave
# [2],[3] ? [1] ==> n_no_predictions_slave++, n_lost_predictions++
# [2] ? [2] ==> n_predictions_improve++ or n_predictions_decrease++ or n_same_predictions++
# [2] ? [3] ==> n_bad_predictions++
# [3] ? [2] ==> n_bad_predictions++
# [3] ? [3] ==> n_predictions_improve++ or n_predictions_decrease++ or n_same_predictions++
#
#####################################################################################


class Test(unittest.TestCase):

    def setUp(self):
        temporary_dir = tempfile.gettempdir()
        # Generate data
        columns = ['c'+str(i) for i in range(0, 10)]

        # Master/Slave data :
        master_data, slave_data, self.res = Test._generate_data()

        master_df = pd.DataFrame(data=master_data, columns=columns)
        master_df.insert(0, 'img', ['img_' + str(i) for i in range(master_data.shape[0])])
        # print master_df.shape, master_df.head(n=master_data.shape[0])

        slave_df = pd.DataFrame(data=slave_data, columns=columns)
        slave_df.insert(0, 'img', ['img_' + str(i) for i in range(master_data.shape[0])])
        # print slave_df.head(n=slave_data.shape[0])

        # Write files
        self.master_file = os.path.join(temporary_dir, "master_submission.csv")
        master_df.to_csv(self.master_file, index=False)
        self.slave_file = os.path.join(temporary_dir, "slave_submission.csv")
        slave_df.to_csv(self.slave_file, index=False)

    def tearDown(self):
        # Run submissions_compare
        os.remove(self.master_file)
        os.remove(self.slave_file)

    def test_submissions_compare(self):
        # self.assertTrue(False)
        threshold = 0.6
        id_threshold = 0.1
        res_ = main(self.master_file, self.slave_file, threshold, id_threshold)

        print res_
        print self.res

        assert (np.array(self.res) == np.array(res_)).all()

    @staticmethod
    def _generate_data():
        master_data = np.zeros((13, 10))
        slave_data = np.zeros((13, 10))

        # res columns :
        # n_same_predictions, n_new_predictions, n_lost_predictions, n_predictions_improve,
        # n_predictions_decrease, n_bad_predictions, n_no_predictions_master, n_no_predictions_slave
        res = [0, 0, 0, 0, 0, 0, 0, 0]

        # [1] ? [1] ==> n_no_predictions_master++, n_no_predictions_slave++
        master_data[0, :] = 0.5
        slave_data[0, :] = 0.5
        res[6] += 1; res[7] += 1
        # [1] ? [2] ==> n_no_predictions_master++, n_new_predictions++ by slave
        master_data[1, :] = 0.5
        slave_data[1, :] = 0.1/9.0; slave_data[1, 0] = 0.9
        res[6] += 1; res[1] += 1
        # [1] ? [3] ==> n_no_predictions_master++, n_new_predictions++ by slave
        master_data[2, :] = 0.5
        slave_data[2, :] = (1.0 - 0.63)/9.0; slave_data[2, 1] = 0.63
        res[6] += 1; res[1] += 1
        # [2] ? [1] ==> n_no_predictions_slave++, n_lost_predictions++
        master_data[3, :] = 0.1/9.0; master_data[3, 2] = 0.9
        slave_data[3, :] = 0.5
        res[7] += 1; res[2] += 1
        # [3] ? [1] ==> n_no_predictions_slave++, n_lost_predictions++
        master_data[4, :] = (1.0 - 0.76)/9.0; master_data[4, 3] = 0.76
        slave_data[4, :] = 0.5
        res[7] += 1; res[2] += 1
        # [2] ? [2] -> improve
        master_data[5, :] = (1.0 - 0.63)/9.0; master_data[5, 4] = 0.63
        slave_data[5, :] = (1.0 - 0.8)/9.0; slave_data[5, 4] = 0.80
        res[3] += 1
        # [2] ? [2] -> decrease
        master_data[6, :] = (1.0 - 0.83)/9.0; master_data[6, 5] = 0.83
        slave_data[6, :] = (1.0 - 0.63)/9.0; slave_data[6, 5] = 0.63
        res[4] += 1
        # [2] ? [2] -> identical
        master_data[7, :] = (1.0 - 0.73)/9.0; master_data[7, 6] = 0.73
        slave_data[7, :] = (1.0 - 0.71)/9.0; slave_data[7, 6] = 0.71
        res[0] += 1
        # [2] ? [3]
        master_data[8, :] = (1.0 - 0.73)/9.0; master_data[8, 7] = 0.73
        slave_data[8, :] = (1.0 - 0.71)/9.0; slave_data[8, 8] = 0.71
        res[5] += 1
        # [3] ? [2]
        master_data[9, :] = (1.0 - 0.66)/9.0; master_data[9, 9] = 0.66
        slave_data[9, :] = (1.0 - 0.78)/9.0; slave_data[9, 0] = 0.78
        res[5] += 1
        # [3] ? [3] -> improve
        master_data[10, :] = (1.0 - 0.66)/9.0; master_data[10, 1] = 0.66
        slave_data[10, :] = (1.0 - 0.78)/9.0; slave_data[10, 1] = 0.78
        res[3] += 1
        # [3] ? [3] -> decrease
        master_data[11, :] = (1.0 - 0.86)/9.0; master_data[11, 2] = 0.86
        slave_data[11, :] = (1.0 - 0.68)/9.0; slave_data[11, 2] = 0.68
        res[4] += 1
        # [3] ? [3] -> identical
        master_data[12, :] = (1.0 - 0.81)/9.0; master_data[12, 3] = 0.81
        slave_data[12, :] = (1.0 - 0.85)/9.0; slave_data[12, 3] = 0.85
        res[0] += 1

        return master_data, slave_data, res


if __name__ == "__main__":
    unittest.main()

