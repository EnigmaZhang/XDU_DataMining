import pandas as pd
from efficient_apriori import apriori

"""
@Author: Zhang Xiaotian
@School: Xidian University
@Reference:
This program is to use Apriori algorithm to find high confidence in America parliament vote record.
"""


def read_data(path):
    data = pd.read_table(path, sep=",", header=None, na_values="?")
    print(data)
    col_names = [
        "party",
        "handicapped-infants",
        "water-project-cost-sharing",
        "adoption-of-the-budget-resolution",
        "physician-fee-freeze",
        "el-salvador-aid",
        "religious-groups-in-schools",
        "anti-satellite-test-ban",
        "aid-to-nicaraguan-contras",
        "mx-missile",
        "immigration",
        "synfuels-corporation-cutback",
        "education-spending",
        "superfund-right-to-sue",
        "crime",
        "duty-free-exports",
        "export-administration-act-south-africa",
    ]
    data = data.fillna(0)
    data.columns = col_names
    data = data.replace({"y": 1, "n": -1})
    return data


data = read_data(r"./data/house-votes-84.data")
items = []
for i in range(data.shape[0]):
    items.append(tuple([str(data.values[i][j]) for j in range(data.shape[1])]))
item_sets, rules = apriori(items, min_support=0.3, min_confidence=0.9)
for i in range(len(rules)):
    print(rules[i])
