import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

def load_model(model_path, device):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def segment_image(model, image_path, device):
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")
    orig_image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_image.shape[:2]
    
    # Compute new dimensions (divisible by 32)
    new_h = int(np.ceil(orig_h / 32) * 32)
    new_w = int(np.ceil(orig_w / 32) * 32)
    
    # Compute padding for height and width
    pad_top = (new_h - orig_h) // 2
    pad_bottom = new_h - orig_h - pad_top
    pad_left = (new_w - orig_w) // 2
    pad_right = new_w - orig_w - pad_left
    
    # Pad the image (using black padding)
    padded_image = cv2.copyMakeBorder(orig_image_rgb, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=0)
    
    # Preprocess: normalize and convert to tensor (C x H x W)
    image = padded_image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # From HWC to CHW
    image_tensor = torch.tensor(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        # Apply sigmoid to get probabilities
        prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        # Threshold the probabilities to get a binary mask
        binary_mask = (prob_mask > 0.5).astype(np.uint8) * 255
    
    # Crop the binary mask back to the original image size
    binary_mask_cropped = binary_mask[pad_top:pad_top+orig_h, pad_left:pad_left+orig_w]
    
    return orig_image, binary_mask_cropped

def measure_bracket(binary_mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Assume the largest contour is the bracket
    largest_contour = max(contours, key=cv2.contourArea)
    # Compute bounding box (x, y, width, height)
    x, y, width, height = cv2.boundingRect(largest_contour)
    return (x, y, width, height), largest_contour

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "unet_bracket.pth"
    model = load_model(model_path, device)
    
    test_image_path = "test_img.jpeg"  
    orig_image, binary_mask = segment_image(model, test_image_path, device)
    
    result = measure_bracket(binary_mask)
    if result is None:
        print("No bracket detected in the image.")
    else:
        bbox, contour = result  
        x, y, width, height = bbox
    
    DPI = 96 
    # Convert to mm
    length_mm = (height * 25.4) / DPI
    breadth_mm = (width * 25.4) / DPI

    # Draw bounding box and display length & breadth in mm
    cv2.rectangle(orig_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(orig_image, f"L: {length_mm:.2f}mm, B: {breadth_mm:.2f}mm", 
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save results
    cv2.imwrite("bracket_detection_mm.png", orig_image)
    print(f"Detected Bracket - Length: {length_mm:.2f}mm, Breadth: {breadth_mm:.2f}mm")
    print("Results saved as bracket_detection_mm.png")

    # Draw bounding box and display length & breadth
    cv2.rectangle(orig_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(orig_image, f"L: {height}px, B: {width}px", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save results
    cv2.imwrite("segmented_mask.png", binary_mask)
    cv2.imwrite("bracket_detection.png", orig_image)
    print(f"Detected Bracket - Length: {height}px, Breadth: {width}px")
    print("Results saved as segmented_mask.png and bracket_detection.png")
