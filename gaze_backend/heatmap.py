import numpy as np
import matplotlib.pyplot as plt

# Combine metrics into matrix
metrics_custom = np.array([
    rouge1_custom,
    rougeL_custom,
    cosine_custom
])

metrics_static = np.array([
    rouge1_static,
    rougeL_static,
    cosine_static
])

# Compute correlation matrices
corr_custom = np.corrcoef(metrics_custom)
corr_static = np.corrcoef(metrics_static)

labels = ["ROUGE-1", "ROUGE-L", "Cosine"]

# ---------- Heatmap: Gaze-based ----------
plt.figure()
plt.imshow(corr_custom)
plt.colorbar()
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)
plt.title("Correlation Heatmap (Gaze-based T5)")
plt.show()

# ---------- Heatmap: Static ----------
plt.figure()
plt.imshow(corr_static)
plt.colorbar()
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)
plt.title("Correlation Heatmap (Static T5)")
plt.show()
