import matplotlib.pyplot as plt

log_loss_a = [1.46, 0.80, 0.71, 0.73, 0.65, 0.65, 0.64, 0.63]
log_loss_b = [1.50, 0.82, 0.70, 0.67, 0.68, 0.63, 0.63, 0.63]

log_acc_a = [0.55, 0.71, 0.79, 0.77, 0.80, 0.79, 0.80, 0.81]
log_acc_b = [0.57, 0.71, 0.77, 0.79, 0.77, 0.79, 0.79, 0.79]

log_rec_a = [0.56, 0.72, 0.78, 0.75, 0.79, 0.77, 0.78, 0.78]
log_rec_b = [0.56, 0.69, 0.76, 0.77, 0.75, 0.77, 0.76, 0.78]

log_prec_a = [0.58, 0.73, 0.80, 0.78, 0.82, 0.79, 0.82, 0.81]
log_prec_b = [0.56, 0.75, 0.78, 0.80, 0.78, 0.79, 0.81, 0.80]

knn_acc_a = [0.42, 0.62, 0.70, 0.71, 0.70, 0.67, 0.68, 0.71]
knn_acc_b = [0.38, 0.62, 0.69, 0.69, 0.65, 0.73, 0.71, 0.70]

knn_rec_a = [0.39, 0.59, 0.67, 0.67, 0.65, 0.63, 0.64, 0.67]
knn_rec_b = [0.36, 0.59, 0.64, 0.65, 0.62, 0.70, 0.69, 0.67]

knn_prec_a = [0.42, 0.68, 0.71, 0.75, 0.71, 0.68, 0.67, 0.75]
knn_prec_b = [0.40, 0.65, 0.72, 0.72, 0.65, 0.79, 0.73, 0.75]

x = [0, 10, 20, 30, 40, 50, 60, 100]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

ax1, ax2, ax3, ax4 = axes.flatten()

ax1.figure(figsize=(10, 10))
ax1.plot(x, log_loss_a, marker="o", color="blue")
ax1.plot(x, log_loss_b, marker="o", linestyle="dashed", color="red")
ax1.legend(["Filtered", "Blurred"])
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Logistic loss")

ax2.figure(figsize=(10, 10))
ax2.plot(x, log_acc_a, marker="o", color="blue")
ax2.plot(x, log_acc_b, marker="o", linestyle="dashed", color="red")
ax2.legend(["Filtered", "Blurred"])
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Logistic accuracy")

ax3.figure(figsize=(10, 10))
ax3.plot(x, log_rec_a, marker="o", color="blue")
ax3.plot(x, log_rec_b, marker="o", linestyle="dashed", color="red")
ax3.legend(["Filtered", "Blurred"])
ax3.set_xlabel("Epochs")
ax3.set_ylabel("Logistic recall")

ax4.figure(figsize=(10, 10))
ax4.plot(x, log_prec_a, marker="o", color="blue")
ax4.plot(x, log_prec_b, marker="o", linestyle="dashed", color="red")
ax4.legend(["Filtered", "Blurred"])
ax4.set_xlabel("Epochs")
ax4.set_ylabel("Logistic precision")

fig.suptitle("Logistic regression metrics")

plt.show()

fig, axes = plt.subplots(1, 3, figsize=(10, 10))

ax1, ax2, ax3 = axes.flatten()

ax1.plot(x, knn_acc_a, marker="o", color="blue")
ax1.plot(x, knn_acc_b, marker="o", linestyle="dashed", color="red")
ax1.legend(["Filtered", "Blurred"])
ax1.set_xlabel("Epochs")
ax1.set_ylabel("KNN accuracy")

ax2.plot(x, knn_rec_a, marker="o", color="blue")
ax2.plot(x, knn_rec_b, marker="o", linestyle="dashed", color="red")
ax2.legend(["Filtered", "Blurred"])
ax2.set_xlabel("Epochs")
ax2.set_ylabel("KNN recall")

ax3.plot(x, knn_prec_a, marker="o", color="blue")
ax3.plot(x, knn_prec_b, marker="o", linestyle="dashed", color="red")
ax3.legend(["Filtered", "Blurred"])
ax3.set_xlabel("Epochs")
ax3.set_ylabel("KNN precision")

fig.suptitle("KNN metrics")

plt.show()

colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

plt.figure(figsize=(10, 10))
for i, _list in enumerate([log_acc_a, log_rec_a, log_prec_a]):
    plt.plot(x, _list, marker="o", color=colors[i])
for i, _list in enumerate([log_acc_b, log_rec_b, log_prec_b]):
    plt.plot(x, _list, marker="o", linestyle="dashed", color=colors[i])

plt.legend(["log_loss_b", "log_acc_a", "log_acc_b", "log_rec_a", "log_rec_b", "log_prec_a", "log_prec_b"])

plt.xlabel("Epochs")
plt.ylabel("Logistic loss")
plt.show()

colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

plt.figure(figsize=(10, 10))
for i, _list in enumerate([knn_acc_a, knn_rec_a, knn_prec_a]):
    plt.plot(x, _list, marker="o", color=colors[i])
for i, _list in enumerate([knn_acc_b, knn_rec_b, knn_prec_b]):
    plt.plot(x, _list, marker="o", linestyle="dashed", color=colors[i])

plt.legend(["knn_acc_a", "knn_rec_a", "knn_prec_a", "knn_acc_b", "knn_rec_b", "knn_prec_b"])

plt.xlabel("Epochs")
plt.ylabel("Logistic loss")
plt.show()