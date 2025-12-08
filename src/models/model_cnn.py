#a CNN for identifying green red yellow
#architecture: Conv2d->teLU->maxpool
#flatten->linear->reLU->droput->linear

import torch.nn as nn

class TrafficCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )



    #forward pass thhrough the CNN, retursn the logits for each class so N * num_classes
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
