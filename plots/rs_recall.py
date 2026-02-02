import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


csv_file = '/home/ztl/cl/LLM_Partition/results/mlp_trainer/serial_metrics.csv' 
data = pd.read_csv(csv_file)

recall_2 = data['Recall@2'].to_numpy()
recall_3 = data['Recall@3'].to_numpy()
recall_4 = data['Recall@4'].to_numpy()

x_ticks = np.array(range(0, len(recall_2), 1))

plt.plot(x_ticks, recall_2, label='Recall@2')
plt.plot(x_ticks, recall_3, label='Recall@3')
plt.plot(x_ticks, recall_4, label='Recall@4')

plt.xticks(x_ticks)

plt.xlabel('Layer index')
plt.ylabel('Recall')
plt.title('Expert Prediction based on intermediate feature of Layer 0')
plt.legend()
plt.tight_layout()
plt.grid(True, alpha=0.3)


SAVE_DIR = "/home/ztl/cl/LLM_Partition/results/mlp_trainer"
img_path = os.path.join(SAVE_DIR, "expert_prediction.png")

plt.savefig(img_path, dpi=300)