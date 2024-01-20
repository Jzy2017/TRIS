import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace

def plot_laplace(mu=0, b=1):
    x = np.linspace(laplace.ppf(0.01, mu, b), laplace.ppf(0.99, mu, b), 1000)
    plt.figure(figsize=(12, 8))
    plt.plot(x, laplace.pdf(x, mu, b), 'r-', label='Laplace')
    plt.title("Laplace Distribution")
    plt.xlabel("X")
    plt.ylabel("Probability Density")
    plt.text(0.6, 0.8, '$\mu$ = {}, b = {}'.format(mu, b), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
    plt.legend()
    plt.show()

# 使用默认参数μ=0, b=1绘制拉普拉斯分布图
plot_laplace()