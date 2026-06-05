import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import json

from data_loader import FixationDataset
from fixation_model import FixationNet


def create_and_save_val_image(batch, pred, epoch, save_dir="outputs"):
    """
    Takes the raw batch dictionary and the model's logit predictions,
    formats them into grids, and saves a single 3-row image.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Extract and format the tensors
    raw_images = batch['raw_image'].permute(0, 3, 1, 2).float() / 255.0
    fixations = batch['fixation'].cpu()
    prediction_normalized = torch.sigmoid(pred).cpu()

    # 2. Make the grids (forcing 16 columns)
    grid_raw = make_grid(raw_images, nrow=16)
    grid_fix = make_grid(fixations, nrow=16)
    grid_pred = make_grid(prediction_normalized, nrow=16)

    # 3. Convert to numpy arrays for matplotlib
    img_raw = np.asarray(TF.to_pil_image(grid_raw.detach()))
    img_fix = np.asarray(TF.to_pil_image(grid_fix.detach()))
    img_pred = np.asarray(TF.to_pil_image(grid_pred.detach()))

    # 4. Plot and save
    fig, axs = plt.subplots(nrows=3, ncols=1, squeeze=False, figsize=(20, 10))

    axs[0, 0].imshow(img_raw)
    axs[0, 0].set_title("Raw Input Images")

    axs[1, 0].imshow(img_fix, cmap='gray')
    axs[1, 0].set_title("Ground Truth Fixations")

    axs[2, 0].imshow(img_pred, cmap='gray')
    axs[2, 0].set_title("Model Predictions")

    for ax in axs.flat:
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"val_epoch_{epoch:03d}.png"), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    ds_train = FixationDataset(
        root_dir=r'.\cv2_project_data',
        image_file=r'.\cv2_project_data\train_images.txt',
        fixation_file=r'.\cv2_project_data\train_fixations.txt',
        image_transform=v2.Compose([v2.ToTensor(),
                                    v2.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])]),
        fixation_transform=v2.Compose([v2.ToTensor()]),
    )

    ds_val = FixationDataset(
        root_dir=r'.\cv2_project_data',
        image_file=r'.\cv2_project_data\val_images.txt',
        fixation_file=r'.\cv2_project_data\val_fixations.txt',
        image_transform=v2.Compose([v2.ToTensor(),
                                    v2.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])]),
        fixation_transform=v2.Compose([v2.ToTensor()]),
    )

    train_loader = DataLoader(ds_train, batch_size=16, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=64, shuffle=False)

    model = FixationNet().to(device)
    epochs = 100
    loss_func = F.binary_cross_entropy_with_logits
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": []
    }
    best_val_loss = float('inf')
    start_epoch = 0

    # --- CHECKPOINT LOADING LOGIC ---
    checkpoint_path = os.path.join("checkpoints", "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Loading...")
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start at the next epoch
        best_val_loss = checkpoint['val_loss']

        print(f"Resuming training from epoch {start_epoch} with Best Val Loss: {best_val_loss:.4f}")

        # Attempt to load existing metrics so we don't overwrite history
        if os.path.exists("metrics.json"):
            with open("metrics.json", "r") as f:
                metrics = json.load(f)
    else:
        print("No checkpoint found. Starting from scratch.")

    print("Starting Training Loop...")
    for epoch in range(start_epoch, epochs):
        model.train()

        # --- 1. TRAINING PHASE ---
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            xb = batch['image'].to(device)
            yb = batch['fixation'].to(device)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            opt.step()

            total_train_loss += loss.item() * xb.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        # --- 2. VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0.0
        epoch_aucs = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                xb = batch['image'].to(device)
                yb = batch['fixation'].to(device)

                pred = model(xb)
                loss = loss_func(pred, yb)
                total_val_loss += loss.item() * xb.size(0)
                # Using the very first validation batch to generate visual grid progression
                if i == 0:
                    create_and_save_val_image(batch, pred, epoch)

                # --- ROC-AUC BATCH CALCULATION ---
                # Convert predictions to probabilities via Sigmoid, move to CPU numpy
                preds_prob = torch.sigmoid(pred).cpu().numpy()
                gts = yb.cpu().numpy()

                # Calculate ROC-AUC per individual image in the batch
                for j in range(xb.size(0)):
                    # Flatten the 2D map into a 1D vector of pixels
                    y_score = preds_prob[j].flatten()

                    # Binarize the ground-truth density.
                    # Adjust threshold (e.g., 0.5 or 0.1) depending on your dataset preference
                    y_true = (gts[j].flatten() > 0.5).astype(int)

                    # ROC-AUC is mathematically undefined if an image contains only 0s or only 1s
                    if len(np.unique(y_true)) == 2:
                        auc = roc_auc_score(y_true, y_score)
                        epoch_aucs.append(auc)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        avg_val_auc = np.mean(epoch_aucs) if epoch_aucs else 0.0
        print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
              f"| Val ROC-AUC: {avg_val_auc:.4f}")

        # --- 3. METRIC PERSISTENCE & CHECKPOINTING ---
        metrics["train_loss"].append(avg_train_loss)
        metrics["val_loss"].append(avg_val_loss)
        metrics["val_auc"].append(avg_val_auc)

        # Write scores out to JSON
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Keep track of the best performing model weights based on val evaluation
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_loss': best_val_loss,
            }
            torch.save(checkpoint, os.path.join("checkpoints", "best_model.pth"))
            print(f"  --> Saved new best weights! (Val Loss: {best_val_loss:.4f})")