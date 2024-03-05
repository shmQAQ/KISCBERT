import pandas as pd
import numpy as np
df = pd.read_csv('KISCBERT/data/0.5_similarity.csv')

column = df['CSV SMILES']
char_dict = {}

for text in column:
    for char in text:
        if char in char_dict:
            char_dict[char] += 1
        else:
            char_dict[char] = 1
print(max(len(text) for text in column))

for char, count in char_dict.items():
    print(f'Character: {char}, Count: {count}')