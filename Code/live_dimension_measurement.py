import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

#Get model weights
def load_model(model_path, device): 
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

#Generate frames from test video
def segment_frame(model, frame, device):
    orig_h, orig_w = frame.shape[:2]
    new_h = int(np.ceil(orig_h / 32) * 32)
    new_w = int(np.ceil(orig_w / 32) * 32)
    pad_top = (new_h - orig_h) // 2
    pad_bottom = new_h - orig_h - pad_top
    pad_left = (new_w - orig_w) // 2
    pad_right = new_w - orig_w - pad_left
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    padded_frame = cv2.copyMakeBorder(frame_rgb, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=0)
    image = padded_frame.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        binary_mask = (prob_mask > 0.5).astype(np.uint8) * 255
    
    return binary_mask[pad_top:pad_top+orig_h, pad_left:pad_left+orig_w]

#Detect circular cavity
def detect_black_circle(mask):
    height = mask.shape[0]
    top_half = mask[:int(height * 0.5), :]
    inverted_mask = cv2.bitwise_not(top_half)
    kernel = np.ones((3,3), np.uint8)
    cleaned_mask = cv2.morphologyEx(inverted_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_circle, best_circularity = None, 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius * radius
        area_ratio = area / circle_area
        circle_score = circularity * area_ratio
        if circle_score > best_circularity and 0.7 < area_ratio < 1.3:
            best_circularity, best_circle = circle_score, (int(x), int(y), int(radius))

    return best_circle

#Measurement of bracket dimensions
def measure_bracket(binary_mask, frame, dpi=96): #DPI 96 of device
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame, None
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(largest_contour)
    length_mm = (height * 25.4) / dpi
    breadth_mm = (width * 25.4) / dpi
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(frame, f"L: {length_mm:.2f}mm, B: {breadth_mm:.2f}mm",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame, (length_mm, breadth_mm)

# yeh wala har frame me circle measure karega
def measure_circle(frame, mask, dpi=96):
    circle = detect_black_circle(mask)
    if circle:
        x, y, radius = circle
        radius_mm = (radius * 25.4) / dpi
        cv2.circle(frame, (x, y), radius, (255, 0, 0), 2)
        cv2.putText(frame, f"Radius: {radius_mm:.2f}mm", (x - 40, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return frame

#Video capture and applying segmentation model
def live_feed_measurement(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        binary_mask = segment_frame(model, frame, device)
        frame, dimensions = measure_bracket(binary_mask, frame)
        frame = measure_circle(frame, binary_mask)

        cv2.imshow("Live Bracket and Circle Measurement", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "unet_bracket.pth"
    live_feed_measurement(model_path)