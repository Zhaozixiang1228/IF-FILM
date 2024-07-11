import numpy as np
import matplotlib.pyplot as plt

def draw_epoch_metric(input, metric_list, best, best2, savepath=None):
    epochnum, metricnum = input.shape
    x = np.linspace(1, epochnum, epochnum)
    plt.figure(figsize=(16, 8))
    if metricnum <= 12:
        for row in range(3):
            for column in range(4):
                plt.subplot(3, 4, 4 * row + column + 1)
                plt.plot(x, input[:, 4 * row + column])
                plt.title(metric_list[4 * row + column])
                plt.axhline(best[4 * row + column], color='red')
                plt.axhline(best2[4 * row + column], color='blue')
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)
    plt.close('all')