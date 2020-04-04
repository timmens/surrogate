"""Moves files created in final to sciebo folder.

Move old data file to ARCHIVE and update logfile.

Change ``sciebo_path`` to make code run on your computer.
"""
import datetime
import os
import shutil

from bld.project_paths import project_paths_join as ppj

sciebo_path = "/home/tm/sciebo/uni-master/master-thesis/structUncertainty/"


def standard_copy(src, name):
    shutil.copy(src, sciebo_path + name)


def logged_copy(src, name, logfile_prefix):
    logfile_name = logfile_prefix + "_logfile.txt"

    # time stamp
    now = str(datetime.datetime.now())[:19]
    now = now.replace(":", "-").replace(" ", "-")

    # check if logfile exists
    logfile_exists = os.path.isfile(sciebo_path + logfile_name)
    # check if data file exists
    datafile_exists = os.path.isfile(sciebo_path + name)

    # read and write time stamp
    if logfile_exists:
        with open(sciebo_path + logfile_name, "r") as file:
            old_time_stamp = file.readline()
    with open(sciebo_path + logfile_name, "w") as file:
        file.writelines(now)

    if logfile_exists and datafile_exists:
        to_archive_file_name = (
            sciebo_path + "ARCHIVE/" + name + "-" + old_time_stamp + ".pkl"
        )
        # move old file to ARCHIVE
        shutil.move(sciebo_path + name, to_archive_file_name)

    # move new file to sciebo
    shutil.copy(src, sciebo_path + name)


if __name__ == "__main__":

    # full data set ###################################################################
    data_path = ppj("OUT_FINAL", "data_with_predictions.pkl")
    logged_copy(src=data_path, name="data_with_predictions.pkl", logfile_prefix="data")

    # losses of models ################################################################
    losses_path = ppj("OUT_ANALYSIS", "losses.csv")
    standard_copy(losses_path, "losses.csv")

    # bootstrap mae plot ##############################################################
    bootstrap_path = ppj("OUT_FIGURES", "bootstrap_mae.pdf")
    standard_copy(bootstrap_path, "bootstrap_mae.pdf")

    # ridge variable selection ########################################################
    variable_selection_path = ppj("OUT_FIGURES", "ridge_variable_selection.pdf")
    standard_copy(variable_selection_path, "ridge_variable_selection.pdf")
