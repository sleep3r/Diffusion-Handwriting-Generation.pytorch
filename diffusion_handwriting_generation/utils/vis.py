from matplotlib import pyplot as plt
import numpy as np


def show_strokes(
    strokes: np.ndarray,
    name: str = "",
    show_output: bool = True,
    scale: int = 1,
) -> None:
    """Plots strokes into image"""
    positions = np.cumsum(strokes, axis=0).T[:2]
    W, H = np.max(positions, axis=-1) - np.min(positions, axis=-1)

    plt.figure(figsize=(scale * W / H, scale))
    plt.axis("off")

    prev_ind = 0
    for ind, is_end in enumerate(strokes[:, 2]):
        if is_end:
            plt.plot(
                positions[0][prev_ind:ind + 1],
                positions[1][prev_ind:ind + 1],
                color="black",
            )
            prev_ind = ind + 1


    if name:
        plt.savefig("./" + name + ".png", bbox_inches="tight")
    if show_output:
        plt.show()
    else:
        plt.close()


def show_image(**images: np.ndarray) -> None:
    """Plots images in one row"""
    n = len(images)

    for i, (name, image) in enumerate(images.items()):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap="gray")
    plt.show()
