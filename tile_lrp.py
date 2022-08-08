import cv2
import os
import glob
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    path = "visualization/caltech"
    single_model_path = glob.glob(os.path.join(path, "1_*"))[0]
    stacked_model_path = glob.glob(os.path.join(path, "4_vgg*"))[0]

    lrsplit = lambda s, x="_": s.split(x, 1)[1].rsplit(x, 1)[0]

    single_backbone = lrsplit(single_model_path)
    stacked_backbones = lrsplit(stacked_model_path).split("-")

    plt.rcParams["figure.figsize"] = [9.00, 6.00]
    plt.rcParams["figure.autolayout"] = True

    for image_path in glob.glob(os.path.join(single_model_path, "label_*_orig.png")):
        image_name = os.path.basename(image_path)
        tag = image_name.split("_", 1)[1].rsplit("_", 1)[0]

        orig_image = cv2.imread(image_path)
        single_pred_image = cv2.imread(os.path.join(single_model_path, f"pred_{tag}_{single_backbone}.png"))
        single_label_image = cv2.imread(os.path.join(single_model_path, f"label_{tag}_{single_backbone}.png"))

        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        axes[0][0].set_title('Original')
        axes[0][1].set_title(f'Stacked {stacked_backbones[0]}')
        axes[0][2].set_title(f'Stacked {stacked_backbones[1]}')
        axes[1][0].set_title(f'Single {single_backbone}')
        axes[1][1].set_title(f'Stacked {stacked_backbones[2]}')
        axes[1][2].set_title(f'Stacked {stacked_backbones[3]}')
        axes[0][0].imshow(orig_image)
        plt.grid(False)
        plt.axis('off')
        axes[1][0].imshow(single_pred_image)
        plt.grid(False)
        plt.axis('off')

        stacked_images = []
        for i, backbone in enumerate(stacked_backbones):
            tmp = cv2.imread(os.path.join(stacked_model_path, f"{tag}_{backbone}.png"))
            stacked_images.append(tmp)
            axes[(1 if i > 1 else 0)][i + 1 - (2 if i > 1 else 0)].imshow(tmp)
            plt.grid(False)
            plt.axis('off')

        for i in range(2):
            for j in range(3):
                axes[i][j].grid(False)
                axes[i][j].axis('off')

        plt.savefig(os.path.join(path, f"{tag}.png"))
        plt.clf()
        plt.close()
        # fig.tight_layout()
        # fig.subplots_adjust(top=1)
        # fig.savefig(os.path.join(path, f"{tag}.png"))