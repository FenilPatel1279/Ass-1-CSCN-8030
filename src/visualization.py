import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_heatmap(df, save_path="../outputs/correlation_heatmap.png"):
    # Ensure the folder exists before saving
    folder = os.path.dirname(save_path)
    if folder != "":  
        os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(6,4))
    sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
    plt.title("Correlation Heatmap")
    plt.savefig(save_path)
    plt.close()

def plot_fertilizer_vs_yield(df, save_path="../outputs/fertilizer_vs_yield.png"):
    # Ensure the folder exists before saving
    folder = os.path.dirname(save_path)
    if folder != "":  
        os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(6,4))
    sns.scatterplot(x="Fertilizer_kg", y="Yield_kg_ha", data=df)
    plt.title("Fertilizer vs Yield")
    plt.savefig(save_path)
    plt.close()

