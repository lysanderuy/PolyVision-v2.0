import cv2
import numpy as np
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
import dataset

# cv2_imshow function for local display (saves to file)
def cv2_imshow(image, filename="iou_visualization.jpg"):
    """Save image to file (replacement for Colab's cv2_imshow)"""
    cv2.imwrite(filename, image)
    print(f"Image saved as: {filename}")

# You need these variables from your dataset.py
# Make sure dataset.py has been run first, or define them here:
DATA_SET_NAME = "SEAMaP-Binary-Full"  # From your dataset registration
TRAIN_DATA_SET_NAME = f"{DATA_SET_NAME}-train"

# Load your dataset and model as you did in your original code.
metadata = MetadataCatalog.get(TRAIN_DATA_SET_NAME)
dataset_train = DatasetCatalog.get(TRAIN_DATA_SET_NAME)

# Select a sample entry from the dataset
dataset_entry = dataset_train[0]
image = cv2.imread(dataset_entry["file_name"])

# Define IoU thresholds
iou_thresholds = [0.3, 0.5, 0.75, 0.95]

for idx, iou_threshold in enumerate(iou_thresholds):
    # Assuming you have predicted bounding boxes in the same format as ground truth
    # Replace this with your actual predicted bounding boxes
    predicted_boxes = dataset_entry["annotations"]

    # Create a copy of the image for each threshold
    image_with_boxes = image.copy()

    for bbox in predicted_boxes:
        bbox_coords = bbox["bbox"]
        x, y, w, h = bbox_coords

        # Adjust the predicted bounding box based on IoU threshold
        if iou_threshold == 0.95:
            # Close alignment with the ground truth
            adjusted_bbox = bbox_coords
        elif iou_threshold == 0.3:
            # Quite off alignment with the ground truth
            adjusted_bbox = [x + 2, y + 2, w * 1.7, h * 1.8]  # Adjust the coordinates as needed
        elif iou_threshold == 0.5:
            # Adjust as needed for IoU 0.5 (example adjustment)
            adjusted_bbox = [x + 5, y + 5, w * 1.5, h * 1.5]  # Example adjustment for IoU 0.5
        elif iou_threshold == 0.75:
            # Adjust as needed for IoU 0.75 (example adjustment)
            adjusted_bbox = [x - 5, y - 5, w * 1.2, h * 1.2]  # Example adjustment for IoU 0.75
        else:
            # Adjust as needed for other thresholds
            adjusted_bbox = bbox_coords  # Default case

        # Calculate IoU
        intersection = [
            max(x, adjusted_bbox[0]),
            max(y, adjusted_bbox[1]),
            min(x + w, adjusted_bbox[0] + adjusted_bbox[2]),
            min(y + h, adjusted_bbox[1] + adjusted_bbox[3])
        ]
        intersection_area = max(0, intersection[2] - intersection[0]) * max(0, intersection[3] - intersection[1])
        union_area = w * h + adjusted_bbox[2] * adjusted_bbox[3] - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0

        # Draw ground truth bounding box in green
        cv2.rectangle(
            image_with_boxes,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            (0, 255, 0),  # Bounding box color (green)
            2  # Bounding box thickness
        )

        # Draw IoU threshold bounding box in red
        cv2.rectangle(
            image_with_boxes,
            (int(adjusted_bbox[0]), int(adjusted_bbox[1])),
            (int(adjusted_bbox[0] + adjusted_bbox[2]), int(adjusted_bbox[1] + adjusted_bbox[3])),
            (0, 0, 255),  # Bounding box color (red)
            2  # Bounding box thickness
        )

        # Display the IoU value
        cv2.putText(
            image_with_boxes,
            f"IoU: {iou:.2f}",
            (int(x), int(y) - 10),  # Adjust text position
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # Font scale
            (255, 255, 255),  # Text color (white)
            1,  # Text thickness
            cv2.LINE_AA
        )

    # Save the image with both bounding boxes and IoU values
    cv2_imshow(image_with_boxes, f"iou_threshold_{iou_threshold}.jpg")

# Display the original image with ground truth bounding box
visualizer = Visualizer(
    image[:, :, ::-1],
    metadata=metadata,
    scale=0.8,
    instance_mode=ColorMode.IMAGE_BW
)
out = visualizer.draw_dataset_dict(dataset_entry)
cv2_imshow(out.get_image()[:, :, ::-1], "original_ground_truth.jpg")

print("All IoU visualizations have been saved!")
print("Files created:")
print("- iou_threshold_0.3.jpg")
print("- iou_threshold_0.5.jpg") 
print("- iou_threshold_0.75.jpg")
print("- iou_threshold_0.95.jpg")
print("- original_ground_truth.jpg")