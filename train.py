# # train.py — works with folder layout like:
# # data/train/067/, data/train/067_forg/, data/train/068/, data/train/068_forg/, ...

# import os
# import random
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader

# IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# # -------------------------
# # Dataset that reads folder-per-person structure
# # -------------------------
# class SignatureDataset(Dataset):
#     def __init__(self, base_dir, img_size=(150,150)):
#         self.genuine = []
#         self.forged = []
#         self.img_size = img_size

#         if not os.path.isdir(base_dir):
#             raise ValueError(f"Base dir not found: {base_dir}")

#         for entry in sorted(os.listdir(base_dir)):
#             folder = os.path.join(base_dir, entry)
#             if not os.path.isdir(folder):
#                 continue
#             # treat folders containing "forg" (case-insensitive) as forged
#             if "forg" in entry.lower():
#                 target_list = self.forged
#             else:
#                 target_list = self.genuine

#             for fname in os.listdir(folder):
#                 if fname.lower().endswith(IMG_EXTS):
#                     path = os.path.join(folder, fname)
#                     target_list.append(path)

#         # remove any unreadable images
#         self.genuine = [p for p in self.genuine if self._check_img(p)]
#         self.forged = [p for p in self.forged if self._check_img(p)]

#         if len(self.genuine) == 0 or len(self.forged) == 0:
#             raise ValueError(f"Empty data lists: genuine={len(self.genuine)}, forged={len(self.forged)}")

#     def _check_img(self, p):
#         img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
#         return img is not None

#     def __len__(self):
#         # produce roughly balanced number of pairs
#         return min(len(self.genuine), len(self.forged)) * 2

#     def __getitem__(self, idx):
#         if idx % 2 == 0:
#             # positive pair (genuine-genuine)
#             a = random.choice(self.genuine)
#             b = random.choice(self.genuine)
#             label = 0.0
#         else:
#             # negative pair (genuine-forged)
#             a = random.choice(self.genuine)
#             b = random.choice(self.forged)
#             label = 1.0

#         A = cv2.imread(a, cv2.IMREAD_GRAYSCALE)
#         B = cv2.imread(b, cv2.IMREAD_GRAYSCALE)
#         A = cv2.resize(A, self.img_size).astype("float32") / 255.0
#         B = cv2.resize(B, self.img_size).astype("float32") / 255.0
#         A = np.expand_dims(A, axis=0)  # (1,H,W)
#         B = np.expand_dims(B, axis=0)

#         return torch.tensor(A), torch.tensor(B), torch.tensor([label], dtype=torch.float32)

# # -------------------------
# # Simple Siamese model
# # -------------------------
# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 32, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2)
#         )
#         # calculate flatten size for 150x150 -> conv/pool result:
#         # Two conv (kernel5) + two pool(2) -> final spatial approx: ((150-4)//2 -4)//2 = ~34
#         self.fc = nn.Sequential(
#             nn.Linear(64 * 34 * 34, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 128)
#         )

#     def forward_once(self, x):
#         x = self.cnn(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

#     def forward(self, x1, x2):
#         return self.forward_once(x1), self.forward_once(x2)

# # -------------------------
# # Contrastive loss
# # -------------------------
# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super().__init__()
#         self.margin = margin

#     def forward(self, out1, out2, label):
#         dist = nn.functional.pairwise_distance(out1, out2)
#         loss = torch.mean((1 - label) * torch.pow(dist, 2) +
#                           label * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
#         return loss

# # -------------------------
# # Train loop
# # -------------------------
# def main():
#     # <-- change this if your train folder is elsewhere -->
#     base_train_dir = r"C:\\Users\\Ankit Jha\\Downloads\\archive\\sign_data\\sign_data\\train"
#     batch_size = 8
#     epochs = 30
#     lr = 1e-5

#     dataset = SignatureDataset(base_train_dir)
#     print(f"Found genuine={len(dataset.genuine)} forged={len(dataset.forged)}")
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SiameseNetwork().to(device)
#     criterion = ContrastiveLoss()
#     opt = optim.Adam(model.parameters(), lr=lr)

#     model.train()
#     for ep in range(epochs):
#         running = 0.0
#         count = 0
#         for A, B, label in loader:
#             A, B, label = A.to(device), B.to(device), label.to(device)
#             opt.zero_grad()
#             o1, o2 = model(A, B)
#             loss = criterion(o1, o2, label)
#             loss.backward()
#             opt.step()
#             running += loss.item()
#             count += 1
#         print(f"Epoch {ep+1}/{epochs} — avg loss: {running/count:.4f}")

#     os.makedirs("models", exist_ok=True)
#     torch.save(model.state_dict(), os.path.join("models", "siamese.pth"))
#     print("Saved models/siamese.pth")

# if __name__ == "__main__":
#     main()






# import os
# import random
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader

# IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# # ======================================================
# # Dataset for Signature Forgery (genuine/forged folders)
# # ======================================================
# class SignatureDataset(Dataset):
#     def __init__(self, base_dir, img_size=(150,150)):
#         self.genuine = []
#         self.forged = []
#         self.img_size = img_size

#         if not os.path.isdir(base_dir):
#             raise ValueError(f"Base dir not found: {base_dir}")

#         for entry in sorted(os.listdir(base_dir)):
#             folder = os.path.join(base_dir, entry)
#             if not os.path.isdir(folder):
#                 continue
#             if "forg" in entry.lower():   # forged folder
#                 target_list = self.forged
#             else:                         # genuine folder
#                 target_list = self.genuine

#             for fname in os.listdir(folder):
#                 if fname.lower().endswith(IMG_EXTS):
#                     path = os.path.join(folder, fname)
#                     target_list.append(path)

#         # remove unreadable images
#         self.genuine = [p for p in self.genuine if self._check_img(p)]
#         self.forged = [p for p in self.forged if self._check_img(p)]

#         if len(self.genuine) == 0 or len(self.forged) == 0:
#             raise ValueError(f"Empty data lists: genuine={len(self.genuine)}, forged={len(self.forged)}")

#     def _check_img(self, p):
#         img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
#         return img is not None

#     def __len__(self):
#         return min(len(self.genuine), len(self.forged)) * 2

#     def __getitem__(self, idx):
#         if idx % 2 == 0:
#             # positive pair (genuine-genuine)
#             a = random.choice(self.genuine)
#             b = random.choice(self.genuine)
#             label = 0.0
#         else:
#             # negative pair (genuine-forged)
#             a = random.choice(self.genuine)
#             b = random.choice(self.forged)
#             label = 1.0

#         A = cv2.imread(a, cv2.IMREAD_GRAYSCALE)
#         B = cv2.imread(b, cv2.IMREAD_GRAYSCALE)
#         A = cv2.resize(A, self.img_size).astype("float32") / 255.0
#         B = cv2.resize(B, self.img_size).astype("float32") / 255.0
#         A = np.expand_dims(A, axis=0)
#         B = np.expand_dims(B, axis=0)

#         return torch.tensor(A), torch.tensor(B), torch.tensor([label], dtype=torch.float32)


# # ======================================================
# # Dataset for Handwriting Verification (writerID folders)
# # ======================================================
# class HandwritingDataset(Dataset):
#     def __init__(self, base_dir, img_size=(150,150)):
#         self.data = {}
#         self.img_size = img_size

#         if not os.path.isdir(base_dir):
#             raise ValueError(f"Base dir not found: {base_dir}")

#         for writer in sorted(os.listdir(base_dir)):
#             folder = os.path.join(base_dir, writer)
#             if not os.path.isdir(folder):
#                 continue
#             imgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]
#             if len(imgs) > 1:   # need at least 2 samples
#                 self.data[writer] = imgs

#         self.writers = list(self.data.keys())
#         if len(self.writers) < 2:
#             raise ValueError("Need at least 2 writers with >1 samples each")

#     def __len__(self):
#         return 100000  # dynamic sampling

#     def __getitem__(self, idx):
#         if random.random() < 0.5:
#             # positive pair (same writer)
#             w = random.choice(self.writers)
#             a, b = random.sample(self.data[w], 2)
#             label = 0.0
#         else:
#             # negative pair (different writers)
#             w1, w2 = random.sample(self.writers, 2)
#             a = random.choice(self.data[w1])
#             b = random.choice(self.data[w2])
#             label = 1.0

#         A = cv2.imread(a, cv2.IMREAD_GRAYSCALE)
#         B = cv2.imread(b, cv2.IMREAD_GRAYSCALE)
#         A = cv2.resize(A, self.img_size).astype("float32") / 255.0
#         B = cv2.resize(B, self.img_size).astype("float32") / 255.0
#         A = np.expand_dims(A, axis=0)
#         B = np.expand_dims(B, axis=0)

#         return torch.tensor(A), torch.tensor(B), torch.tensor([label], dtype=torch.float32)


# # ======================================================
# # Siamese Network
# # ======================================================
# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 32, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(64 * 34 * 34, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 128)
#         )

#     def forward_once(self, x):
#         x = self.cnn(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

#     def forward(self, x1, x2):
#         return self.forward_once(x1), self.forward_once(x2)


# # ======================================================
# # Contrastive Loss
# # ======================================================
# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super().__init__()
#         self.margin = margin

#     def forward(self, out1, out2, label):
#         dist = nn.functional.pairwise_distance(out1, out2)
#         loss = torch.mean((1 - label) * torch.pow(dist, 2) +
#                           label * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
#         return loss


# # ======================================================
# # Training Functions
# # ======================================================
# def train_signature():
#     base_train_dir = r"C:\Users\Ankit Jha\Downloads\archive\sign_data\sign_data\train"
#     batch_size = 8
#     epochs = 30
#     lr = 1e-5

#     dataset = SignatureDataset(base_train_dir)
#     print(f"[Signature] Found genuine={len(dataset.genuine)} forged={len(dataset.forged)}")
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SiameseNetwork().to(device)
#     criterion = ContrastiveLoss()
#     opt = optim.Adam(model.parameters(), lr=lr)

#     model.train()
#     for ep in range(epochs):
#         running = 0.0
#         count = 0
#         for A, B, label in loader:
#             A, B, label = A.to(device), B.to(device), label.to(device)
#             opt.zero_grad()
#             o1, o2 = model(A, B)
#             loss = criterion(o1, o2, label)
#             loss.backward()
#             opt.step()
#             running += loss.item()
#             count += 1
#         print(f"[Signature] Epoch {ep+1}/{epochs} — avg loss: {running/count:.4f}")

#     os.makedirs("models", exist_ok=True)
#     torch.save(model.state_dict(), os.path.join("models", "siamese_signature.pth"))
#     print("Saved models/siamese_signature.pth")


# def train_handwriting():
#     base_train_dir = r"E:\cvl-database-1-1\cvl-database-1-1\testset\lines"
#     batch_size = 8
#     epochs = 20
#     lr = 1e-4

#     dataset = HandwritingDataset(base_train_dir)
#     print(f"[Handwriting] Found {len(dataset.writers)} writers")
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SiameseNetwork().to(device)
#     criterion = ContrastiveLoss()
#     opt = optim.Adam(model.parameters(), lr=lr)

#     model.train()
#     for ep in range(epochs):
#         running = 0.0
#         count = 0
#         for A, B, label in loader:
#             A, B, label = A.to(device), B.to(device), label.to(device)
#             opt.zero_grad()
#             o1, o2 = model(A, B)
#             loss = criterion(o1, o2, label)
#             loss.backward()
#             opt.step()
#             running += loss.item()
#             count += 1
#         print(f"[Handwriting] Epoch {ep+1}/{epochs} — avg loss: {running/count:.4f}")

#     os.makedirs("models", exist_ok=True)
#     torch.save(model.state_dict(), os.path.join("models", "siamese_handwriting.pth"))
#     print("Saved models/siamese_handwriting.pth")


# # ======================================================
# # Main
# # ======================================================
# if __name__ == "__main__":
#     # Uncomment the one you want to train:
    
#     # train_signature()
#     train_handwriting()





import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ======================================================
# File extensions supported (added .tif)
# ======================================================
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")

# ======================================================
# Dataset for Signature Forgery (genuine/forged folders)
# ======================================================
class SignatureDataset(Dataset):
    def __init__(self, base_dir, img_size=(150,150)):
        self.genuine = []
        self.forged = []
        self.img_size = img_size

        if not os.path.isdir(base_dir):
            raise ValueError(f"Base dir not found: {base_dir}")

        for entry in sorted(os.listdir(base_dir)):
            folder = os.path.join(base_dir, entry)
            if not os.path.isdir(folder):
                continue
            if "forg" in entry.lower():   # forged folder
                target_list = self.forged
            else:                         # genuine folder
                target_list = self.genuine

            for fname in os.listdir(folder):
                if fname.lower().endswith(IMG_EXTS):
                    path = os.path.join(folder, fname)
                    target_list.append(path)

        # remove unreadable images
        self.genuine = [p for p in self.genuine if self._check_img(p)]
        self.forged = [p for p in self.forged if self._check_img(p)]

        if len(self.genuine) == 0 or len(self.forged) == 0:
            raise ValueError(f"Empty data lists: genuine={len(self.genuine)}, forged={len(self.forged)}")

    def _check_img(self, p):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        return img is not None

    def __len__(self):
        return min(len(self.genuine), len(self.forged)) * 2

    def __getitem__(self, idx):
        if idx % 2 == 0:
            # positive pair (genuine-genuine)
            a = random.choice(self.genuine)
            b = random.choice(self.genuine)
            label = 0.0
        else:
            # negative pair (genuine-forged)
            a = random.choice(self.genuine)
            b = random.choice(self.forged)
            label = 1.0

        A = cv2.imread(a, cv2.IMREAD_GRAYSCALE)
        B = cv2.imread(b, cv2.IMREAD_GRAYSCALE)
        A = cv2.resize(A, self.img_size).astype("float32") / 255.0
        B = cv2.resize(B, self.img_size).astype("float32") / 255.0
        A = np.expand_dims(A, axis=0)
        B = np.expand_dims(B, axis=0)

        return torch.tensor(A), torch.tensor(B), torch.tensor([label], dtype=torch.float32)


# ======================================================
# Dataset for Handwriting Verification (writerID folders)
# ======================================================
class HandwritingDataset(Dataset):
    def __init__(self, base_dir, img_size=(150,150)):
        self.data = {}
        self.img_size = img_size

        if not os.path.isdir(base_dir):
            raise ValueError(f"Base dir not found: {base_dir}")

        for writer in sorted(os.listdir(base_dir)):
            folder = os.path.join(base_dir, writer)
            if not os.path.isdir(folder):
                continue
            imgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]
            if len(imgs) > 1:   # need at least 2 samples
                self.data[writer] = imgs

        self.writers = list(self.data.keys())
        print(f"[HandwritingDataset] Writers found: {len(self.writers)}")
        for w in self.writers[:5]:  # show first 5 writers for sanity
            print(f"  {w}: {len(self.data[w])} samples")

        if len(self.writers) < 2:
            raise ValueError("Need at least 2 writers with >1 samples each")

    def __len__(self):
        return 100000  # dynamic sampling

    def __getitem__(self, idx):
        if random.random() < 0.5:
            # positive pair (same writer)
            w = random.choice(self.writers)
            a, b = random.sample(self.data[w], 2)
            label = 0.0
        else:
            # negative pair (different writers)
            w1, w2 = random.sample(self.writers, 2)
            a = random.choice(self.data[w1])
            b = random.choice(self.data[w2])
            label = 1.0

        A = cv2.imread(a, cv2.IMREAD_GRAYSCALE)
        B = cv2.imread(b, cv2.IMREAD_GRAYSCALE)
        A = cv2.resize(A, self.img_size).astype("float32") / 255.0
        B = cv2.resize(B, self.img_size).astype("float32") / 255.0
        A = np.expand_dims(A, axis=0)
        B = np.expand_dims(B, axis=0)

        return torch.tensor(A), torch.tensor(B), torch.tensor([label], dtype=torch.float32)


# ======================================================
# Siamese Network
# ======================================================
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )
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


# ======================================================
# Contrastive Loss
# ======================================================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dist = nn.functional.pairwise_distance(out1, out2)
        loss = torch.mean((1 - label) * torch.pow(dist, 2) +
                          label * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss


# ======================================================
# Training Functions
# ======================================================
def train_signature():
    base_train_dir = r"C:\Users\Ankit Jha\Downloads\archive\sign_data\sign_data\train"
    batch_size = 8
    epochs = 30
    lr = 1e-5

    dataset = SignatureDataset(base_train_dir)
    print(f"[Signature] Found genuine={len(dataset.genuine)} forged={len(dataset.forged)}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
     print(f"[INFO] Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
     print("[INFO] Training on CPU")
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        running = 0.0
        count = 0
        for A, B, label in loader:
            A, B, label = A.to(device), B.to(device), label.to(device)
            opt.zero_grad()
            o1, o2 = model(A, B)
            loss = criterion(o1, o2, label)
            loss.backward()
            opt.step()
            running += loss.item()
            count += 1
        print(f"[Signature] Epoch {ep+1}/{epochs} — avg loss: {running/count:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("models", "siamese_signature.pth"))
    print("Saved models/siamese_signature.pth")


def train_handwriting():
    base_train_dir = r"E:\cvl-database-1-1\cvl-database-1-1\testset\lines"
    batch_size = 8
    epochs = 5
    lr = 1e-4

    dataset = HandwritingDataset(base_train_dir)
    print(f"[Handwriting] Found {len(dataset.writers)} writers")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
     print(f"[INFO] Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
     print("[INFO] Training on CPU")
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        running = 0.0
        count = 0
        for A, B, label in loader:
            A, B, label = A.to(device), B.to(device), label.to(device)
            opt.zero_grad()
            o1, o2 = model(A, B)
            loss = criterion(o1, o2, label)
            loss.backward()
            opt.step()
            running += loss.item()
            count += 1
        print(f"[Handwriting] Epoch {ep+1}/{epochs} — avg loss: {running/count:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("models", "siamese_handwriting.pth"))
    print("Saved models/siamese_handwriting.pth")


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    # Uncomment the one you want to train:

    # train_signature()
    train_handwriting()
