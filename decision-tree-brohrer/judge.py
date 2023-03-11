import numpy as np
import matplotlib.pyplot as plt

class Judge(object):
    def __init__(self, arrival_times_df):
        self.arrival_times_df = arrival_times_df

    def find_total_absolute_deviation(self, cols=None):
        if cols is None:
            eval_set_df = self.arrival_times_df
        else:
            eval_set_df = self.arrival_times_df.loc[:, cols]

        departure_time = self.find_departure_time(eval_set_df)
        actual_arrivals = eval_set_df.loc[
            eval_set_df.index == departure_time, :].values
        total_deviation = np.sum(np.abs(actual_arrivals))

        return total_deviation, departure_time

    def find_departure_time(self, eval_set_df):
        # Find the 90th percentile lateness for each row
        lateness = eval_set_df.quantile(q=.9, axis=1)

        # Find the departure time that corresponds to a lateness of 0,
        # i.e., the one that gets us there on time 90% of the days.
        lateness[lateness > 0] = -120
        i_dep = np.argmax(lateness.values)
        departure_time = eval_set_df.index[i_dep]
        return departure_time

