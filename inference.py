import argparse
import torch
import timm
import segmentation_models_pytorch as smp
import torchvision.transforms as T
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

def pad_to_square(img, fill=0):
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    new_img = Image.new(img.mode, (s, s), fill)
    new_img.paste(img, ((s - w) // 2, (s - h) // 2))
    return new_img


def load_classifier(weights_path):
    model = timm.create_model(
        "convnext_tiny",
        pretrained=False,   # Important
        num_classes=2,
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model



classifier_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_segmentation_model(weights_path):
    model = smp.Unet(
        encoder_name="efficientnet-b5",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model


seg_transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
])


def classify_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img_t = classifier_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_t).argmax(1).item()

    return pred  # 0 = good, 1 = bad


def segment_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = pad_to_square(img)
    img = img.resize((384, 384))

    img_t = seg_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_t))[0, 0].cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8)
    return img, mask


def overlay_mask(img, mask, color=(255, 0, 0), alpha=0.4):
    img = img.convert("RGB")
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(img.size)

    # Create a red layer
    red_layer = Image.new("RGB", img.size, color)

    # Blend only scratch region
    blended = Image.blend(img, red_layer, alpha)

    # Final output: original image where mask=0, blended where mask=1
    blended_np = np.array(blended)
    img_np = np.array(img)
    mask_3ch = np.stack([mask]*3, axis=-1)

    final = np.where(mask_3ch == 1, blended_np, img_np)

    return Image.fromarray(final)



def analyze_image(image_path, classifier, segmenter):
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create output folder
    os.makedirs("output", exist_ok=True)

    label = classify_image(classifier, image_path)

    if label == 0:
        print("Image is GOOD. No scratch detected.")
        return

    print("Image is BAD. Running segmentation...")

    img, mask = segment_image(segmenter, image_path)
    overlay = overlay_mask(img, mask)

    mask_save_path = f"output/{base_name}_mask.png"
    overlay_save_path = f"output/{base_name}_overlay.png"

    # Save only mask & overlay
    Image.fromarray(mask * 255).save(mask_save_path)
    overlay.save(overlay_save_path)

    print(f"Saved mask  {mask_save_path}")
    print(f"Saved overlay  {overlay_save_path}")

    # Display results (NOT saved)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Scratch Detection Pipeline")

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--classifier", type=str, required=True)
    parser.add_argument("--segmenter", type=str, required=True)

    args = parser.parse_args()

    classifier = load_classifier(args.classifier)
    segmenter = load_segmentation_model(args.segmenter)

    analyze_image(args.image, classifier, segmenter)


if __name__ == "__main__":
    main()
