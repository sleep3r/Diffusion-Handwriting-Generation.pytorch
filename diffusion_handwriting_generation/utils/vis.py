from matplotlib import pyplot as plt
import numpy as np


def show(strokes, name="", show_output=True, scale=1):
    positions = np.cumsum(strokes, axis=0).T[:2]
    prev_ind = 0
    W, H = np.max(positions, axis=-1) - np.min(positions, axis=-1)
    plt.figure(figsize=(scale * W / H, scale))

    for ind, value in enumerate(strokes[:, 2]):
        if value > 0.5:
            plt.plot(
                positions[0][prev_ind:ind], positions[1][prev_ind:ind], color="black"
            )
            prev_ind = ind

    plt.axis("off")
    if name:
        plt.savefig("./" + name + ".png", bbox_inches="tight")
    if show_output:
        plt.show()
    else:
        plt.close()
