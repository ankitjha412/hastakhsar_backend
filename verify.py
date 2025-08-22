# import os
# import cv2
# import torch
# import torch.nn as nn
# import numpy as np
# from train import SiameseNetwork, IMG_EXTS

# def preprocess_image(img_path, img_size=(150,150)):
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise ValueError(f"Failed to load image: {img_path}")
#     img = cv2.resize(img, img_size).astype("float32") / 255.0
#     img = np.expand_dims(img, axis=0)  # (1,H,W)
#     img = np.expand_dims(img, axis=0)  # (1,1,H,W)
#     return torch.tensor(img, dtype=torch.float32)

# def verify_signature(model, img1_path, img2_path, device, threshold=0.5):
#     model.eval()
#     with torch.no_grad():
#         img1 = preprocess_image(img1_path).to(device)
#         img2 = preprocess_image(img2_path).to(device)
#         out1, out2 = model(img1, img2)
#         dist = nn.functional.pairwise_distance(out1, out2)
#         dist_val = dist.item()
#         label = "Genuine" if dist_val < threshold else "Forged"
#         return dist_val, label

# def main():
#     import argparse
#     parser = argparse.ArgumentParser(description="Verify signature pairs using Siamese Network")
#     parser.add_argument("model_path", type=str, help="Path to saved model weights")
#     parser.add_argument("img1", type=str, help="Path to first image (genuine)")
#     parser.add_argument("img2", type=str, help="Path to second image (test)")
#     parser.add_argument("--threshold", type=float, default=0.5, help="Distance threshold for genuine/forged")
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SiameseNetwork().to(device)
#     model.load_state_dict(torch.load(args.model_path, map_location=device))

#     verify_signature(model, args.img1, args.img2, device, threshold=args.threshold)

# if __name__ == "__main__":
#     main()





import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from train import SiameseNetwork

# -------------------------
# Preprocess image
# -------------------------
def preprocess_image(img_path, img_size=(150,150)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img = cv2.resize(img, img_size).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # (1,H,W)
    img = np.expand_dims(img, axis=0)  # (1,1,H,W)
    return torch.tensor(img, dtype=torch.float32)

# -------------------------
# Verification function
# -------------------------
def verify_pair(model, img1_path, img2_path, device, threshold=0.5, task="signature"):
    model.eval()
    with torch.no_grad():
        img1 = preprocess_image(img1_path).to(device)
        img2 = preprocess_image(img2_path).to(device)
        out1, out2 = model(img1, img2)
        dist = nn.functional.pairwise_distance(out1, out2)
        dist_val = dist.item()

        if task == "signature":
            label = "Genuine" if dist_val < threshold else "Forged"
        elif task == "handwriting":
            label = "Same Writer" if dist_val < threshold else "Different Writer"
        else:
            raise ValueError(f"Unknown task: {task}")

        return dist_val, label

# -------------------------
# CLI entry
# -------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify pairs using Siamese Network")
    parser.add_argument("model_path", type=str, help="Path to saved model weights (.pth)")
    parser.add_argument("img1", type=str, help="Path to first image")
    parser.add_argument("img2", type=str, help="Path to second image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Distance threshold")
    parser.add_argument("--task", type=str, default="signature", choices=["signature", "handwriting"],
                        help="Verification task: signature or handwriting")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    dist, label = verify_pair(model, args.img1, args.img2, device,
                              threshold=args.threshold, task=args.task)

    print(f"Distance: {dist:.4f} → Prediction: {label}")

if __name__ == "__main__":
    main()
