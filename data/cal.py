import pandas as pd
import numpy as np
df = pd.read_excel('/home/echoshm/chem/bert/data/our_data.xlsx')

column = df['smiles']
char_dict = {}

for text in column:
    for char in text:
        if char in char_dict:
            char_dict[char] += 1
        else:
            char_dict[char] = 1


for char, count in char_dict.items():
    print(f'Character: {char}, Count: {count}')