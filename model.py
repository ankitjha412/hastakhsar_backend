# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super(SiameseNetwork, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 32, 5),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 5),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(64 * 33 * 33, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 128)
#         )

#     def forward_once(self, x):
#         x = self.cnn(x)
#         x = x.view(x.size()[0], -1)
#         x = self.fc(x)
#         return x

#     def forward(self, img1, img2):
#         out1 = self.forward_once(img1)
#         out2 = self.forward_once(img2)
#         return F.pairwise_distance(out1, out2)




import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )
        # For 150x150 → conv/pool result ≈ 34x34
        self.fc = nn.Sequential(
            nn.Linear(64 * 34 * 34, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)
