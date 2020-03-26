import datetime
import shutil

from bld.project_paths import project_paths_join as ppj

if __name__ == "__main__":
    # check for equivalence and move old file to archive if not equivalent

    # move data frame to shared sciebo folder (only works on computer of Tim Mensinger)
    data_path = ppj("OUT_DATA", "data_with_predictions.pkl")
    sciebo_path = "/home/tm/sciebo/uni-master/master-thesis/structUncertainty/"

    now = str(datetime.datetime.now())[:19]
    now = now.replace(":", "-").replace(" ", "-")
    file_name = "data_with_predictions-" + str(now) + ".pkl"

    shutil.copy(data_path, sciebo_path + file_name)
