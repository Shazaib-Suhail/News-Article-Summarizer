import pandas as pd
from gensim.summarization import summarize
import numpy as np

# Input File Path
df = pd.read_csv('\\ADM_Files\\Input.csv', sep=',', header=0)
#print(df.head())

# Removal of special character 
df['content'] = df['content'].replace('(','')
df['content'] = df['content'].replace(')','')
df['content'] = df['content'].replace('?','')
df['content'] = df['content'].replace('~','')

#Change the i value to change input (0 < i < 20)
i = 1

#Summarizer Instantiation 
oc = str(df.content[i])
hypo = summarize(str(df.content[i]))

# Reference Summaries
with open("ADM_Files\\Reference_Summary\\summary"+ str(i+1) +".txt") as f:
    contents = f.read()
ref = str(contents)

#Rouge Instance for Evaluation
r = Rouge()
[p, r, f] = r.rouge_l([hypo], [ref])

# Output
print("Original Content")
print(oc)
print("System Generated Summary")
print(hypo)
print("Reference Summary")
print(ref)
print("Evaluation Scores")
print(p,r,f)