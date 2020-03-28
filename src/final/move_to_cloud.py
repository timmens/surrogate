"""Move old data file to ARCHIVE and update logfile.

Change ``sciebo_path`` to make code run on your computer.
"""
import datetime
import os
import shutil

from bld.project_paths import project_paths_join as ppj

if __name__ == "__main__":

    data_path = ppj("OUT_DATA", "data_with_predictions.pkl")
    sciebo_path = "/home/tm/sciebo/uni-master/master-thesis/structUncertainty/"

    # time stamp
    now = str(datetime.datetime.now())[:19]
    now = now.replace(":", "-").replace(" ", "-")

    # check if logfile exists
    logfile_exists = os.path.isfile(sciebo_path + "data_logfile.txt")
    # check if data file exists
    datafile_exists = os.path.isfile(sciebo_path + "data_with_predictions.pkl")

    # read and write time stamp
    if logfile_exists:
        with open(sciebo_path + "data_logfile.txt", "r") as file:
            old_time_stamp = file.readline()
    with open(sciebo_path + "data_logfile.txt", "w") as file:
        file.writelines(now)

    file_name = "data_with_predictions.pkl"

    if logfile_exists and datafile_exists:
        to_archive_file_name = (
            sciebo_path + "ARCHIVE/data_with_predictions-" + old_time_stamp + ".pkl"
        )
        # move old file to ARCHIVE
        shutil.move(sciebo_path + file_name, to_archive_file_name)

    # move new file to sciebo
    shutil.copy(data_path, sciebo_path + file_name)
