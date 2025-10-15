
import os
import cv2
import json
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# ==========================================================
# CONFIG
# ==========================================================
THRESHOLD = 0.3   # Detection confidence threshold

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def process_image_detr(processor, model, image, threshold=0.3, draw_boxes=False):
    """Detect signatures in a single image using Conditional DETR."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image_rgb.shape[:2]])  # (H, W)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    sig_count = 0
    boxes_list = []
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if int(label) == 0 and score > threshold:
            sig_count += 1
            boxes_list.append(box.tolist())
            
            if draw_boxes:
                # Draw green bounding box
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add confidence score label
                label_text = f"Sig: {score:.2f}"
                cv2.putText(image, label_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return sig_count, boxes_list, image if draw_boxes else None


def mirror_horizontal(image):
    """Return a horizontally flipped image."""
    return cv2.flip(image, 1)


def mirror_vertical(image):
    """Return a vertically flipped image."""
    return cv2.flip(image, 0)


def mirror_both(image):
    """Return a horizontally and vertically flipped image."""
    return cv2.flip(image, -1)


def split_and_detect(processor, model, image, threshold=0.3, draw_boxes=False):
    """Split image into left and right halves, detect separately, then merge."""
    h, w = image.shape[:2]
    mid = w // 2
    
    # Split image
    left_half = image[:, :mid].copy()
    right_half = image[:, mid:].copy()
    
    print(f"   📐 Split image: Left ({left_half.shape[1]}px) | Right ({right_half.shape[1]}px)")
    
    # Detect on both halves
    left_count, left_boxes, left_annotated = process_image_detr(
        processor, model, left_half, threshold, draw_boxes
    )
    right_count, right_boxes, right_annotated = process_image_detr(
        processor, model, right_half, threshold, draw_boxes
    )
    
    total_count = left_count + right_count
    print(f"   ✅ Left: {left_count} | Right: {right_count} | Total: {total_count}")
    
    # Merge back
    if draw_boxes:
        merged = np.hstack([left_annotated, right_annotated])
    else:
        merged = image
    
    return total_count, merged


def update_json_file(json_path, updated_counts):
    """Update page signature counts in the JSON file."""
    with open(json_path, 'r') as file:
        data = json.load(file)

    data.update(updated_counts)

    with open(json_path, 'w') as file:
        json.dump(data, file, indent=2)

    print("\n✅ JSON file updated successfully.")


# ==========================================================
# MAIN LOGIC
# ==========================================================
def recheck_low_signature_pages(base_folder):
    """
    Check JSON and re-process pages with signature count < 2.
    Tries multiple strategies:
    1. Horizontal flip
    2. Vertical flip
    3. Both flips
    4. Split image in half and detect separately
    
    Args:
        base_folder: Folder containing `processed_images`, `signature_detected`, and `output.json`
    """

    json_path = os.path.join(base_folder, "output.json")
    processed_dir = os.path.join(base_folder, "processed_images")

    if not os.path.exists(json_path):
        print(json_path)
        print("❌ output.json not found.")
        return

    # Load JSON
    with open(json_path, 'r') as file:
        page_counts = json.load(file)

    print("\n🔍 Checking for pages with less than 2 signatures...")
    pages_to_recheck = [page for page, count in page_counts.items() if count < 2]

    if not pages_to_recheck:
        print("✅ All pages have at least 2 signatures. No recheck needed.")
        return

    print(f"⚠️ Pages to recheck: {pages_to_recheck}")

    # Load model once
    print("\n📦 Loading Conditional DETR model...")
    processor = AutoImageProcessor.from_pretrained("tech4humans/conditional-detr-50-signature-detector")
    model = AutoModelForObjectDetection.from_pretrained("tech4humans/conditional-detr-50-signature-detector")
    model.eval()
    print("✅ Model loaded successfully.")

    updated_counts = {}

    for page in pages_to_recheck:
        page_num = int(page.split("_")[1])
        image_path = os.path.join(processed_dir, f"processed_page_{page_num}.jpg")

        if not os.path.exists(image_path):
            print(f"⚠️ Skipping {page}: image not found.")
            continue

        print(f"\n{'='*60}")
        print(f"🔄 Rechecking {page} (current count: {page_counts[page]})")
        print(f"{'='*60}")

        # Load original image
        original_image = cv2.imread(image_path)
        best_count = page_counts[page]
        best_image = original_image.copy()
        best_method = "original"

        # Strategy 1: Horizontal Flip
        print("\n🔄 Strategy 1: Horizontal Flip")
        h_flipped = mirror_horizontal(original_image)
        count, _, annotated = process_image_detr(
            processor, model, h_flipped.copy(), THRESHOLD, draw_boxes=True
        )
        print(f"   ➡️ Detected: {count} signatures")
        if count > best_count:
            best_count = count
            best_image = annotated
            best_method = "horizontal_flip"

        # Strategy 2: Vertical Flip
        print("\n🔄 Strategy 2: Vertical Flip")
        v_flipped = mirror_vertical(original_image)
        count, _, annotated = process_image_detr(
            processor, model, v_flipped.copy(), THRESHOLD, draw_boxes=True
        )
        print(f"   ➡️ Detected: {count} signatures")
        if count > best_count:
            best_count = count
            best_image = annotated
            best_method = "vertical_flip"

        # Strategy 3: Both Flips
        print("\n🔄 Strategy 3: Horizontal + Vertical Flip")
        both_flipped = mirror_both(original_image)
        count, _, annotated = process_image_detr(
            processor, model, both_flipped.copy(), THRESHOLD, draw_boxes=True
        )
        print(f"   ➡️ Detected: {count} signatures")
        if count > best_count:
            best_count = count
            best_image = annotated
            best_method = "both_flips"

        # Strategy 4: Split Detection (on original)
        print("\n🔄 Strategy 4: Split & Detect (Original)")
        count, merged = split_and_detect(
            processor, model, original_image.copy(), THRESHOLD, draw_boxes=True
        )
        if count > best_count:
            best_count = count
            best_image = merged
            best_method = "split_original"

        # Strategy 5: Split Detection (on horizontal flip)
        print("\n🔄 Strategy 5: Split & Detect (Horizontal Flip)")
        count, merged = split_and_detect(
            processor, model, h_flipped.copy(), THRESHOLD, draw_boxes=True
        )
        if count > best_count:
            best_count = count
            best_image = merged
            best_method = "split_h_flip"

        # Strategy 6: Split Detection (on vertical flip)
        print("\n🔄 Strategy 6: Split & Detect (Vertical Flip)")
        count, merged = split_and_detect(
            processor, model, v_flipped.copy(), THRESHOLD, draw_boxes=True
        )
        if count > best_count:
            best_count = count
            best_image = merged
            best_method = "split_v_flip"

        # Update if improvement found
        if best_count > page_counts[page]:
            print(f"\n✅ IMPROVEMENT FOUND!")
            print(f"   Method: {best_method}")
            print(f"   Count: {page_counts[page]} → {best_count}")
            updated_counts[page] = best_count
            cv2.imwrite(image_path, best_image)
        else:
            print(f"\n❌ No improvement for {page}, keeping original.")

    # Update JSON file with new counts
    if updated_counts:
        update_json_file(json_path, updated_counts)
    else:
        print("\n⚠️ No updates made to JSON file (no improvement detected).")

    print("\n🎯 Recheck completed successfully.")


# ==========================================================
# RUN SCRIPT
# ==========================================================
if __name__ == "__main__":
    # Example: folder where previous detection results exist
    base_folder = r"C:\Users\admin\Desktop\HFS_Signature_Detection\SANCTION LETTER 1"
    base_folder = r"C:\Users\admin\Desktop\HFS_Signature_Detection\DRF 1"
    recheck_low_signature_pages(base_folder)