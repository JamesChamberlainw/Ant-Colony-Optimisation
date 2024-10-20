import numpy as np 
import matplotlib.pyplot as plt
import itertools
import csv
import pandas as pd

def load_data():
    df = pd.DataFrame(columns=['weight', 'value'])

    with open('BankProblem.txt','r') as f:
        reader = csv.reader(f, delimiter = '|')
        rows = list(reader)
        rows.pop(0) # drop the first row (header)
        # select next three rows

        while rows != []:
            rows.pop(0) # drop bag number as its not needed
            weight = float(rows[0][0].split(":")[1].strip())
            rows.pop(0) # drop weight
            value = float(rows[0][0].split(":")[1].strip())
            rows.pop(0) # drop value
            new_df = pd.DataFrame([[weight, value]], columns=['weight', 'value'])
            df = pd.concat([df, new_df], ignore_index=False)

    return df

df = load_data()
