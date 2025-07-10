import pandas as pd
import matplotlib.pyplot as plt

print("Loading Height-Weight dataset")
df = pd.read_csv('hw_200.csv')
print("\nFirst 5 rows:")
print(df.head())

print("\nMean of each column:")
print(df.mean(numeric_only=True))

print("\nMedian of each column:")
print(df.median(numeric_only=True))

print("\nStandard Deviation of each column:")
print(df.std(numeric_only=True))

print("\nPlotting histograms")
df.hist(figsize=(10, 5), bins=20)
plt.tight_layout()
plt.show()
