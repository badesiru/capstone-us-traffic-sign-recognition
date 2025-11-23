import matplotlib.pyplot as plt
import torch
from data_loader import get_dataloaders, CLASSES

def main():
    train_loader, val_loader = get_dataloaders(batch_size=8, num_workers=0)

    images, labels = next(iter(train_loader))

    plt.figure(figsize=(12,4))
    for i in range(8):
        img = images[i].permute(1, 2, 0)
        img = (img * 0.5 + 0.5).numpy()

        plt.subplot(2, 4, i+1)
        plt.imshow(img)
        plt.title(CLASSES[labels[i]])
        plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
