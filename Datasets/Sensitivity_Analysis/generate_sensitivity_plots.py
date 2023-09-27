import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style('darkgrid')

df = pd.read_csv('./kdd_10%_step_sensitivity.csv')
df.sort_values(by=['Steps'], inplace=True)
plt.figure(figsize=(6, 3))
plt.plot(df['Steps']/1500*100, df['RMSE'], '.', label='Lowest RMSE', ms=10)
plt.xlabel('Stride Length (% of block)')
plt.ylabel('RMSE')
plt.title('KDD 10% Step Sensitivity')
plt.tight_layout()
plt.savefig('./kdd_step_sensitivity.pdf', dpi=300)
plt.figure(figsize=(6, 3))
plt.plot(df['Size'], df['RMSE'], '.-', label='Lowest RMSE', ms=10)
plt.xscale('log')
plt.xlabel('Augmented Training Size')
plt.ylabel('RMSE')
plt.title('Augmentation Sensitivity: KDD with 10% Missing Data')
plt.tight_layout()
plt.savefig('./kdd_size_sensitivity.pdf', dpi=300)
plt.figure(figsize=(6, 3))
plt.plot(df['Size'], df['Steps']/1500*100, '.', label='% of block', ms=10)
plt.xscale('log')
plt.xlabel('Augmented Training Size')
plt.ylabel('Stride Length (% of block)')
plt.title('Augmented Size vs Stride: KDD with 10% Missing Data')
plt.tight_layout()
plt.savefig('./kdd_size_vs_step.pdf', dpi=300)
