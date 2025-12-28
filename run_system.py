
import cv2
import torch
import numpy as np
import time
import sys
import os
from ultralytics import YOLO # type: ignore
from torchvision import transforms

# Add the model directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from model_architecture import UNetGenerator 

# --- CONFIGURATION ---
VIDEO_SOURCE = '' # Use 0 for Webcam
ENHANCER_WEIGHTS = ''
DETECTOR_WEIGHTS = ''
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. HELPER: CONTRAST BOOSTER (CLAHE) ---
# This is the secret sauce we added earlier. It removes the "White Haze".
def contrast_booster(image_rgb):
    # Convert to LAB (Lightness/Color)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to Lightness
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge and return
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final

def run_pipeline():
    print(f"--- Starting Pipeline on {DEVICE} ---")

    # 2. LOAD MODELS
    print("Loading Enhancer...")
    enhancer = UNetGenerator().to(DEVICE)
    enhancer.load_state_dict(torch.load(ENHANCER_WEIGHTS, map_location=DEVICE, weights_only=True))
    enhancer.eval()

    print("Loading Detector...")
    detector = YOLO(DETECTOR_WEIGHTS)

    # 3. SETUP TRANSFORM (Must match Training!)
    # We added Normalize((0.5...), (0.5...)) in training. We MUST do it here too.
    ai_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    # Get original video size for display
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- STEP A: PRE-PROCESSING ---
        # Resize for AI
        frame_resized = cv2.resize(frame, (256, 256))
          # Convert BGR (OpenCV) -> RGB (AI)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Apply Transform (ToTensor + Normalize)
        # We need to convert numpy array to PIL Image first for transforms to work easily
        from PIL import Image
        pil_img = Image.fromarray(frame_rgb)
        img_tensor = ai_transform(pil_img).unsqueeze(0).to(DEVICE)  # type: ignore[attr-defined]

        # --- STEP B: ENHANCEMENT ---
        with torch.no_grad():
            clean_tensor = enhancer(img_tensor)

        # --- STEP C: POST-PROCESSING (Convert tensor to image) ---
        clean_img = clean_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        
        # Denormalize: [-1, 1] -> [0, 1]
        clean_img = (clean_img + 1) / 2.0
        clean_img = np.clip(clean_img, 0.0, 1.0) 
        
        # Convert to uint8 [0, 255]
        clean_img_uint8 = (clean_img * 255).astype(np.uint8)
        
        # Apply Contrast Booster (Remove Haze)
        final_enhanced_rgb = contrast_booster(clean_img_uint8)
        
        # Convert RGB to BGR for YOLO/OpenCV
        final_enhanced_bgr = cv2.cvtColor(final_enhanced_rgb, cv2.COLOR_RGB2BGR)
        final_enhanced_bgr = cv2.cvtColor(final_enhanced_rgb, cv2.COLOR_RGB2BGR)

        # --- STEP B: UPSCALE FOR YOLO (CRITICAL FIX) ---
        # YOLO needs a bigger image (e.g., 640x640) to see small mines.
        # We resize the 256x256 Enhanced Image -> 640x640
        yolo_input = cv2.resize(final_enhanced_rgb, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        
        # Convert to BGR for YOLO
        yolo_input_bgr = cv2.cvtColor(yolo_input, cv2.COLOR_RGB2BGR)

        # --- STEP D: DETECTION ---
        # conf=0.4: Lowered threshold slightly to catch objects
        results = detector(yolo_input_bgr, conf=0.25, verbose=False)
        # --- STEP E: VISUALIZATION ---
        annotated_frame = results[0].plot()

        # Resize both to same size for display (e.g. 512x512)
        display_size = (640, 480)
        view_raw = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
        view_enhanced = cv2.resize(annotated_frame, display_size, interpolation=cv2.INTER_CUBIC)


        # Stitch Side-by-Side
        final_display = cv2.hconcat([view_raw, view_enhanced])

        # Draw FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        # Draw Labels (Make them big and visible)
        cv2.putText(final_display, "RAW INPUT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(final_display, "ENHANCED + AI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(final_display, f"FPS: {fps:.1f}", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow('Maritime Security System', final_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()