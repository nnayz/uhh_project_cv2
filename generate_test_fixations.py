import os
import re

import imageio.v2 as imageio
import torch
from torchvision.transforms import v2
from torchvision.transforms import ConvertImageDtype

from fixation_model import FixationNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    test_dir = r"./cv2_project_data/images/test"
    output_dir = r"cv2_project_data/fixation_imcomp/test"
    checkpoint_path = r"./checkpoints/best_model.pth"

    os.makedirs(output_dir, exist_ok=True)

    image_transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model = FixationNet().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_files = sorted(
        f for f in os.listdir(test_dir)
        if f.endswith(".png")
    )

    to_uint8 = ConvertImageDtype(torch.uint8)

    with torch.no_grad():
        for filename in image_files:

            image = imageio.imread(
                os.path.join(test_dir, filename)
            )

            x = image_transform(image)
            x = x.unsqueeze(0).to(device)

            pred = model(x)
            pred = torch.sigmoid(pred)
            pred = to_uint8(pred)

            pred = pred.squeeze().cpu().numpy()

            assert pred.shape == (224, 224)

            image_id = re.search(
                r"image-(\d+)\.png",
                filename
            ).group(1)

            output_filename = f"prediction-{image_id}.png"

            imageio.imwrite(
                os.path.join(output_dir, output_filename),
                pred
            )

    print(f"Saved {len(image_files)} predictions to {output_dir}")


if __name__ == "__main__":
    main()
