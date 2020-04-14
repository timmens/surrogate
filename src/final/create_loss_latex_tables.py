import pandas as pd

from bld.project_paths import project_paths_join as ppj


def main():
    df = pd.read_csv(ppj("OUT_FINAL", "losses_mae_tidy.csv"))

    df = df.query("features==27")
    df = df.rename({"samples": "n"}, axis=1)

    methods = df.method.unique()

    for m in methods:
        df_tmp = df.query(f"method=='{m}'")[["n", "mae"]].set_index("n").T
        df_tmp.round(decimals=4).to_latex(ppj("OUT_TABLES", f"mae_{m}.tex"))


if __name__ == "__main__":
    main()
