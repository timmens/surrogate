"""Moves files created in final to sciebo folder.

Move old data file to ARCHIVE and update logfile.

Change ``sciebo_path`` to make code run on your computer.
"""
import datetime
import os
import shutil

from bld.project_paths import project_paths_join as ppj
from src.utilities.utilities import load_implemented_model_names

sciebo_path = "/home/tm/sciebo/uni-master/master-thesis/structUncertainty/"


def standard_copy(src, name, type_):
    fpath = sciebo_path + type_ + "/"
    shutil.copy(src, fpath + name)


def logged_copy(src, name, logfile_prefix, type_):
    fpath = sciebo_path + type_ + "/"

    logfile_name = logfile_prefix + "_logfile.txt"

    # time stamp
    now = str(datetime.datetime.now())[:19]
    now = now.replace(":", "-").replace(" ", "-")

    # check if logfile exists
    logfile_exists = os.path.isfile(fpath + logfile_name)
    # check if data file exists
    datafile_exists = os.path.isfile(fpath + name)

    # read and write time stamp
    if logfile_exists:
        with open(fpath + logfile_name, "r") as file:
            old_time_stamp = file.readline()
    with open(fpath + logfile_name, "w") as file:
        file.writelines(now)

    if logfile_exists and datafile_exists:
        to_archive_file_name = (
            sciebo_path + "ARCHIVE/" + name + "-" + old_time_stamp + ".pkl"
        )
        # move old file to ARCHIVE
        shutil.move(fpath + name, to_archive_file_name)

    # move new file to sciebo
    shutil.copy(src, fpath + name)


def main():
    # data sets (predicted, testing, training)
    data_path = ppj("OUT_ANALYSIS", "df_prediction.pkl")
    logged_copy(
        data_path, "df_prediction.pkl", logfile_prefix="data_pred", type_="data"
    )

    standard_copy(
        ppj("OUT_DATA", "df_validation.pkl"), "df_validation.pkl", type_="data"
    )

    standard_copy(ppj("OUT_DATA", "df_training.pkl"), "df_training.pkl", type_="data")

    # losses of models
    standard_copy(ppj("OUT_ANALYSIS", "losses.csv"), "losses.csv", type_="data")

    # tidy mae losses
    standard_copy(
        ppj("OUT_FINAL", "losses_mae_tidy.csv"), "losses_mae_tidy.csv", type_="data"
    )

    # bootstrap mae plot
    standard_copy(
        ppj("OUT_FIGURES", "bootstrap_mae.pdf"), "bootstrap_mae.pdf", type_="figures"
    )

    # ridge variable selection
    standard_copy(
        ppj("OUT_FIGURES", "ridge_variable_selection.pdf"),
        "ridge_variable_selection.pdf",
        type_="figures",
    )

    # tidy mae losses plot
    standard_copy(ppj("OUT_FIGURES", "mae_plot.pdf"), "mae_plot.pdf", type_="figures")

    # latex tables
    model_names = load_implemented_model_names()
    for model in model_names:
        standard_copy(
            ppj("OUT_TABLES", f"mae_{model}.tex"), f"mae_{model}.tex", type_="tex"
        )


if __name__ == "__main__":
    main()
