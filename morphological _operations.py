import cv2
import matplotlib.pyplot as plt
import numpy as np

def perform_morphological_operations(image_path):
  try:
    # Load grayscale image
    img = cv2.imread(image_path, 0)

    # Create structuring element
    se = np.ones((3, 3), dtype="uint8")

    # Morphological operations
    eroded_img = cv2.erode(img, se)
    dilated_img = cv2.dilate(img, se)
    opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
    closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)

    # Hit-or-Miss Transform
    se_neg = cv2.bitwise_not(se)
    hit_or_miss_img = cv2.morphologyEx(img, cv2.MORPH_HITMISS, se_neg)

    # Create Matplotlib figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    # Original image
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    # Eroded image
    axes[0, 1].imshow(eroded_img, cmap="gray")
    axes[0, 1].set_title("Eroded")
    axes[0, 1].axis("off")

    # Dilated image
    axes[0, 2].imshow(dilated_img, cmap="gray")
    axes[0, 2].set_title("Dilated")
    axes[0, 2].axis("off")

    # Opened image
    axes[1, 0].imshow(opened_img, cmap="gray")
    axes[1, 0].set_title("Opened")
    axes[1, 0].axis("off")

    # Closed image
    axes[1, 1].imshow(closed_img, cmap="gray")
    axes[1, 1].set_title("Closed")
    axes[1, 1].axis("off")

    # Hit-or-Miss image
    axes[1, 2].imshow(hit_or_miss_img, cmap="gray")
    axes[1, 2].set_title("Hit-or-Miss")
    axes[1, 2].axis("off")

    # Adjust layout
    fig.suptitle("Morphological Operations Results", fontsize=14)
    plt.tight_layout()

  except (FileNotFoundError, cv2.error) as e:
    print("Error:", e)

# Replace with your actual image path
image_path = "images/covid19_data_samples/08.jpeg"
perform_morphological_operations(image_path)

plt.show()