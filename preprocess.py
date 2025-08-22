# import cv2
# import numpy as np

# def preprocess_image(img_path):
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (150, 150))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img
 



import cv2
import numpy as np
import torch
import torch.nn.functional as F

def preprocess_image(img_path, img_size=(150,150)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img = cv2.resize(img, img_size).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)   # (1,H,W)
    img = np.expand_dims(img, axis=0)   # (1,1,H,W)
    return torch.tensor(img, dtype=torch.float32)

def verify_pair(model, img1_path, img2_path, device, threshold=0.5, task="signature"):
    model.eval()
    with torch.no_grad():
        img1 = preprocess_image(img1_path).to(device)
        img2 = preprocess_image(img2_path).to(device)
        out1, out2 = model(img1, img2)
        dist = F.pairwise_distance(out1, out2)
        dist_val = dist.item()

        if task == "signature":
            label = "Genuine" if dist_val < threshold else "Forged"
        elif task == "handwriting":
            label = "Same Writer" if dist_val < threshold else "Different Writer"
        else:
            raise ValueError(f"Unknown task: {task}")

        return dist_val, label
