# Dimension-Analysis-of-Metal-Bracket
Computer Vision System to measure dimensions of metal parts

## Data augmentation and Masking
Image Augmentation for Dataset Expansion – The script applies various augmentations (rotation, brightness contrast, noise, motion blur, and perspective transformation) to enhance the diversity of an image dataset, helping improve machine learning model robustness.

Masking for Feature Isolation – It converts images to grayscale, applies a binary threshold, and uses the mask to extract relevant features, ensuring augmentations focus on specific areas of interest rather than background noise.

Batch Processing and Automated Saving – It processes all .jpeg images in a specified folder, generating 10 augmented versions per image, and saves them with structured filenames in an output directory for easy dataset management.

## Live Dimension Measurement
Real-Time Bracket and Circle Detection – This script uses a U-Net model to segment brackets in a live video feed and detects circular cavities in each frame. It processes frames from a webcam to analyze objects dynamically.

Measurement in Millimeters – It calculates the dimensions (length, breadth, and radius) of detected objects by converting pixel measurements to millimeters based on a 96 DPI assumption, displaying real-world sizes in the video feed.

Live Video Processing with OpenCV – The script continuously processes video frames, applies segmentation and measurement, and overlays detected features (bounding boxes and circles) in real-time, providing visual feedback until the user exits with 'q'.

## Final Detection
Image Segmentation for Bracket Detection – This script uses a pre-trained U-Net model to segment brackets in images by generating a binary mask. The mask isolates the bracket from the background.

Measurement in Pixels and Millimeters – The script extracts the largest detected contour (assumed to be the bracket), computes its bounding box, and converts its dimensions from pixels to millimeters based on a 96 DPI assumption.

Visualization and Output – It overlays the detected bounding box on the original image, annotates the dimensions, and saves the results as images (bracket_detection.png, segmented_mask.png, etc.) for further analysis.
