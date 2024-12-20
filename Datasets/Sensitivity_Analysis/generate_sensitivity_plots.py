import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

font = {"family": "normal", "size": 11}

matplotlib.rc("font", **font)
sns.set_style("darkgrid")

df = pd.read_csv("./kdd_10%_step_sensitivity.csv")
df.sort_values(by=["Steps"], inplace=True)
plt.figure(figsize=(6, 3))
plt.plot(df["Steps"] / 1500 * 100, df["RMSE"], ".-", label="Lowest RMSE", ms=10)
plt.xlabel("Stride Length (% of block)")
plt.ylabel("RMSE")
plt.title("Air Quality Index Imputation 10% Step Sensitivity")
plt.tight_layout()
plt.savefig("./kdd_step_sensitivity.pdf", dpi=300)
plt.figure(figsize=(6, 3))
plt.plot(df["Size"], df["RMSE"], ".-", label="Lowest RMSE", ms=10)
plt.xscale("log")
plt.xlabel("Augmented Samples")
plt.ylabel("RMSE")
plt.title(
    "Augmentation Sensitivity: Air Quality Index Imputation with 10% Missing Data"
)
plt.tight_layout()
plt.savefig("./kdd_size_sensitivity.pdf", dpi=300)
plt.figure(figsize=(6, 3))
plt.plot(df["Size"], df["Steps"] / 1500 * 100, ".-", label="% of block", ms=10)
plt.xscale("log")
plt.xlabel("Augmented Training Size")
plt.ylabel("Stride Length (% of block)")
plt.title(
    "Augmented Size vs Stride: Air Quality Index Imputation with 10% Missing Data"
)
plt.tight_layout()
plt.savefig("./kdd_size_vs_step.pdf", dpi=300)
plt.figure(figsize=(6, 3))
plt.plot(df["Size"] / 1500, df["RMSE"], ".-", label="Lowest RMSE", ms=10)
plt.xscale("log")
plt.xlabel("Augmented Samples per Block")
plt.ylabel("RMSE")
plt.title(
    "Augmentation Sensitivity: Air Quality Index Imputation with 10% Missing Data"
)
plt.tight_layout()
plt.savefig("./kdd_size_sensitivity_per_sample.pdf", dpi=300)
