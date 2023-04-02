""" Convert root file to csv using uproot """

# import time
# from pprint import pprint
# import pickle
import uproot
import pandas as pd


def main():
    """read and convert root to csv file"""
    file_root = uproot.open("~/Desktop/myThesisWork/smOnAMPT/AnaRes1020.root")
    # pprint(file_root.keys())
    # pprint(file_root.values())
    dfs = []
    tree = file_root["fDBEvtTree"]
    # t0 = time.time()
    for key in tree.keys():
        print(key, tree[key].array())
        # print(key, len(tree[key].array()))
        print(key)

        value = tree[key].array()
        data_frame = pd.DataFrame({key: value})
        dfs.append(data_frame)

    # dataframe = pd.concat(dfs, axis=1)
    # t1 = time.time()
    # print(t1-t0)
    # print(dataframe)
    # t2 = time.time()

    # dataframe.to_csv("amptsm.csv")


if __name__ == "__main__":
    main()
