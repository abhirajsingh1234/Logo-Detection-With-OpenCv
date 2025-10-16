# version 5



# import os
# import cv2
# import json
# import torch
# import numpy as np
# import time
# from transformers import AutoImageProcessor, AutoModelForObjectDetection

# # ==========================================================
# # CONFIG
# # ==========================================================
# THRESHOLD = 0.3   # Detection confidence threshold

# # ==========================================================
# # HELPER FUNCTIONS
# # ==========================================================
# def process_image_detr(processor, model, image, threshold=0.3, draw_boxes=False):
#     """Detect signatures in a single image using Conditional DETR."""
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     inputs = processor(images=image_rgb, return_tensors="pt")

#     with torch.no_grad():
#         outputs = model(**inputs)

#     target_sizes = torch.tensor([image_rgb.shape[:2]])
#     results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

#     sig_count = 0
#     boxes_list = []
    
#     for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#         if int(label) == 0 and score > threshold:
#             sig_count += 1
#             boxes_list.append(box.tolist())
            
#             if draw_boxes:
#                 x1, y1, x2, y2 = map(int, box.tolist())
#                 cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 label_text = f"Sig: {score:.2f}"
#                 cv2.putText(image, label_text, (x1, y1 - 10),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     return sig_count, boxes_list, image if draw_boxes else None


# def mirror_horizontal(image):
#     """Return a horizontally flipped image."""
#     return cv2.flip(image, 1)


# def split_and_detect(processor, model, image, threshold=0.3, draw_boxes=False):
#     """Split image into left and right halves, detect separately."""
#     h, w = image.shape[:2]
#     mid = w // 2
    
#     left_half = image[:, :mid].copy()
#     right_half = image[:, mid:].copy()
    
#     left_count, _, left_annotated = process_image_detr(
#         processor, model, left_half, threshold, draw_boxes
#     )
#     right_count, _, right_annotated = process_image_detr(
#         processor, model, right_half, threshold, draw_boxes
#     )
    
#     total_count = left_count + right_count
#     print(f"   L:{left_count} R:{right_count} = {total_count}", end="")
    
#     if draw_boxes:
#         merged = np.hstack([left_annotated, right_annotated])
#     else:
#         merged = image
    
#     return total_count, merged


# def detect_in_regions_batch(processor, model, image, threshold=0.3, draw_boxes=False, region_type='edges'):
#     """
#     Optimized batch detection in regions (edges or corners).
#     Processes all regions and returns best result.
#     """
#     h, w = image.shape[:2]
    
#     if region_type == 'edges':
#         # Try both 30% and 40% in one go
#         configs = [
#             ('30%', 0.3),
#             ('40%', 0.4)
#         ]
#     else:  # corners
#         configs = [
#             ('40%', 0.4),
#             ('50%', 0.5)
#         ]
    
#     best_count = 0
#     best_image = image.copy() if draw_boxes else image
#     best_config = None
    
#     for config_name, size in configs:
#         if region_type == 'edges':
#             edge_h = int(h * size)
#             edge_w = int(w * size)
            
#             regions = {
#                 't': image[0:edge_h, :].copy(),
#                 'b': image[h-edge_h:h, :].copy(),
#                 'l': image[:, 0:edge_w].copy(),
#                 'r': image[:, w-edge_w:w].copy()
#             }
#             coords = {
#                 't': (0, edge_h, 0, w),
#                 'b': (h-edge_h, h, 0, w),
#                 'l': (0, h, 0, edge_w),
#                 'r': (0, h, w-edge_w, w)
#             }
#         else:  # corners
#             corner_h = int(h * size)
#             corner_w = int(w * size)
            
#             regions = {
#                 'TL': image[0:corner_h, 0:corner_w].copy(),
#                 'TR': image[0:corner_h, w-corner_w:w].copy(),
#                 'BL': image[h-corner_h:h, 0:corner_w].copy(),
#                 'BR': image[h-corner_h:h, w-corner_w:w].copy()
#             }
#             coords = {
#                 'TL': (0, corner_h, 0, corner_w),
#                 'TR': (0, corner_h, w-corner_w, w),
#                 'BL': (h-corner_h, h, 0, corner_w),
#                 'BR': (h-corner_h, h, w-corner_w, w)
#             }
        
#         total_count = 0
#         result_image = image.copy() if draw_boxes else image
#         detected = []
        
#         for name, region_img in regions.items():
#             count, _, annotated = process_image_detr(
#                 processor, model, region_img, threshold, draw_boxes
#             )
#             total_count += count
            
#             if draw_boxes and annotated is not None:
#                 y1, y2, x1, x2 = coords[name]
#                 result_image[y1:y2, x1:x2] = annotated
            
#             if count > 0:
#                 detected.append(f"{name}:{count}")
        
#         if total_count > best_count:
#             best_count = total_count
#             best_image = result_image
#             best_config = config_name
    
#     config_str = f"({best_config})" if best_config else ""
#     print(f"   {best_count} sigs {config_str}", end="")
    
#     return best_count, best_image


# def update_json_file(json_path, updated_counts):
#     """Update page signature counts in the JSON file."""
#     with open(json_path, 'r') as file:
#         data = json.load(file)

#     data.update(updated_counts)

#     with open(json_path, 'w') as file:
#         json.dump(data, file, indent=2)

#     print("\n✅ JSON updated")


# # ==========================================================
# # MAIN LOGIC (QUALITY-PRESERVED OPTIMIZATION)
# # ==========================================================
# def recheck_low_signature_pages(base_folder):
#     """
#     Optimized without compromising quality.
#     Runs all strategies but with batch processing and reduced logging.
#     """
    
#     # Search for output.json
#     possible_json_paths = [
#         os.path.join(base_folder, "output.json"),
#         os.path.join(base_folder, "signature_detected", "output.json"),
#     ]
    
#     json_path = None
#     for path in possible_json_paths:
#         if os.path.exists(path):
#             json_path = path
#             break
    
#     if json_path is None:
#         print("❌ output.json not found")
#         return
    
#     print(f"✅ output.json found")
    
#     # Determine directories
#     json_dir = os.path.dirname(json_path)
#     parent_dir = os.path.dirname(json_dir) if os.path.basename(json_dir) == "signature_detected" else json_dir
    
#     # Search for processed_images
#     possible_processed_dirs = [
#         os.path.join(parent_dir, "processed_images"),
#         os.path.join(json_dir, "processed_images"),
#     ]
    
#     processed_dir = None
#     for path in possible_processed_dirs:
#         if os.path.exists(path):
#             processed_dir = path
#             break
    
#     if processed_dir is None:
#         print("❌ processed_images not found")
#         return
    
#     print(f"✅ processed_images found")

#     # Load JSON
#     with open(json_path, 'r') as file:
#         page_counts = json.load(file)

#     pages_to_recheck = [page for page, count in page_counts.items() if count < 2]

#     if not pages_to_recheck:
#         print("✅ All pages have ≥2 signatures")
#         return

#     print(f"\n🔍 Rechecking {len(pages_to_recheck)} pages")

#     # Load model once
#     print("📦 Loading model...", end=" ")
#     processor = AutoImageProcessor.from_pretrained("tech4humans/conditional-detr-50-signature-detector")
#     model = AutoModelForObjectDetection.from_pretrained("tech4humans/conditional-detr-50-signature-detector")
#     model.eval()
#     print("Done ✅")

#     updated_counts = {}
#     total_time = 0
#     page_times = []

#     for idx, page in enumerate(pages_to_recheck, 1):
#         page_num = int(page.split("_")[1])
#         image_path = os.path.join(processed_dir, f"processed_page_{page_num}.jpg")

#         if not os.path.exists(image_path):
#             continue

#         print(f"\n[{idx}/{len(pages_to_recheck)}] {page} (curr:{page_counts[page]})")
        
#         page_start = time.time()
#         original_image = cv2.imread(image_path)
#         best_count = page_counts[page]
#         best_image = original_image.copy()
#         best_method = "original"

#         # Strategy 1: Split Detection
#         print("  Split:", end=" ")
#         count, merged = split_and_detect(processor, model, original_image.copy(), THRESHOLD, True)
#         if count > best_count:
#             best_count, best_image, best_method = count, merged, "split"
#         print()

#         # Strategy 2: Edge Detection (tests both 30% and 40%)
#         print("  Edges:", end=" ")
#         count, edges_img = detect_in_regions_batch(processor, model, original_image.copy(), THRESHOLD, True, 'edges')
#         if count > best_count:
#             best_count, best_image, best_method = count, edges_img, "edges"
#         print()

#         # Strategy 3: Corner Detection (tests both 40% and 50%)
#         print("  Corners:", end=" ")
#         count, corners_img = detect_in_regions_batch(processor, model, original_image.copy(), THRESHOLD, True, 'corners')
#         if count > best_count:
#             best_count, best_image, best_method = count, corners_img, "corners"
#         print()

#         # Strategy 4: Corners on Horizontal Flip
#         print("  Corners-HFlip:", end=" ")
#         h_flipped = mirror_horizontal(original_image)
#         count, corners_img = detect_in_regions_batch(processor, model, h_flipped.copy(), THRESHOLD, True, 'corners')
#         if count > best_count:
#             best_count, best_image, best_method = count, corners_img, "corners_hflip"
#         print()

#         elapsed = time.time() - page_start
#         total_time += elapsed
#         page_times.append((page, elapsed))

#         # Update if improved
#         if best_count > page_counts[page]:
#             print(f"  ✅ {page_counts[page]}→{best_count} ({best_method}) [{elapsed:.1f}s]")
#             updated_counts[page] = best_count
#             cv2.imwrite(image_path, best_image)
#         else:
#             print(f"  ❌ No improvement [{elapsed:.1f}s]")

#     # Update JSON
#     if updated_counts:
#         update_json_file(json_path, updated_counts)

#     # Timing summary
#     print("\n" + "="*50)
#     print("⏱️  TIMING SUMMARY")
#     print("="*50)
#     if page_times:
#         print(f"Total: {total_time:.1f}s ({total_time/60:.1f}min)")
#         print(f"Average: {total_time/len(page_times):.1f}s/page")
#         print(f"Fastest: {min(page_times, key=lambda x: x[1])[1]:.1f}s")
#         print(f"Slowest: {max(page_times, key=lambda x: x[1])[1]:.1f}s")

#     print("\n🎯 Done!")


# # ==========================================================
# # RUN SCRIPT
# # ==========================================================
# if __name__ == "__main__":
#     base_folder = r"C:\Users\admin\Desktop\HFS_Signature_Detection\SL - 694000"
#     base_folder = r"C:\Users\admin\Desktop\HFS_Signature_Detection\sanction latter english"
    
#     recheck_low_signature_pages(base_folder)





#version 6 (final version)


import os
import cv2
import json
import torch
import numpy as np
import time
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

    target_sizes = torch.tensor([image_rgb.shape[:2]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    sig_count = 0
    boxes_list = []
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if int(label) == 0 and score > threshold:
            sig_count += 1
            boxes_list.append(box.tolist())
            
            # if draw_boxes:
            #     x1, y1, x2, y2 = map(int, box.tolist())
            #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     label_text = f"Sig: {score:.2f}"
            #     cv2.putText(image, label_text, (x1, y1 - 10),
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return sig_count, boxes_list, image if draw_boxes else None


def mirror_horizontal(image):
    """Return a horizontally flipped image."""
    return cv2.flip(image, 1)


def split_and_detect(processor, model, image, threshold=0.3, draw_boxes=False):
    """Split image into left and right halves, detect separately."""
    h, w = image.shape[:2]
    mid = w // 2
    
    left_half = image[:, :mid].copy()
    right_half = image[:, mid:].copy()
    
    left_count, _, left_annotated = process_image_detr(
        processor, model, left_half, threshold, draw_boxes
    )
    right_count, _, right_annotated = process_image_detr(
        processor, model, right_half, threshold, draw_boxes
    )
    
    total_count = left_count + right_count
    print(f"   L:{left_count} R:{right_count} = {total_count}", end="")
    
    if draw_boxes:
        merged = np.hstack([left_annotated, right_annotated])
    else:
        merged = image
    
    return total_count, merged


def detect_in_regions_batch(processor, model, image, threshold=0.3, draw_boxes=False, region_type='edges', early_stop=False):
    """
    Optimized batch detection in regions (edges or corners).
    Processes all regions and returns best result.
    Early stops at 3 signatures if early_stop=True.
    """
    h, w = image.shape[:2]
    
    if region_type == 'edges':
        # Try both 30% and 40% in one go
        configs = [
            ('30%', 0.3),
            ('40%', 0.4)
        ]
    else:  # corners
        configs = [
            ('40%', 0.4),
            ('50%', 0.5)
        ]
    
    best_count = 0
    best_image = image.copy() if draw_boxes else image
    best_config = None
    
    for config_name, size in configs:
        # Early stop if we already found 3+ signatures
        if early_stop and best_count >= 3:
            break
            
        if region_type == 'edges':
            edge_h = int(h * size)
            edge_w = int(w * size)
            
            regions = {
                't': image[0:edge_h, :].copy(),
                'b': image[h-edge_h:h, :].copy(),
                'l': image[:, 0:edge_w].copy(),
                'r': image[:, w-edge_w:w].copy()
            }
            coords = {
                't': (0, edge_h, 0, w),
                'b': (h-edge_h, h, 0, w),
                'l': (0, h, 0, edge_w),
                'r': (0, h, w-edge_w, w)
            }
        else:  # corners
            corner_h = int(h * size)
            corner_w = int(w * size)
            
            regions = {
                'TL': image[0:corner_h, 0:corner_w].copy(),
                'TR': image[0:corner_h, w-corner_w:w].copy(),
                'BL': image[h-corner_h:h, 0:corner_w].copy(),
                'BR': image[h-corner_h:h, w-corner_w:w].copy()
            }
            coords = {
                'TL': (0, corner_h, 0, corner_w),
                'TR': (0, corner_h, w-corner_w, w),
                'BL': (h-corner_h, h, 0, corner_w),
                'BR': (h-corner_h, h, w-corner_w, w)
            }
        
        total_count = 0
        result_image = image.copy() if draw_boxes else image
        detected = []
        
        for name, region_img in regions.items():
            # Early stop per region if we hit 3 signatures
            if early_stop and total_count >= 3:
                break
                
            count, _, annotated = process_image_detr(
                processor, model, region_img, threshold, draw_boxes
            )
            total_count += count
            
            # if draw_boxes and annotated is not None:
            #     y1, y2, x1, x2 = coords[name]
            #     result_image[y1:y2, x1:x2] = annotated
            
            if count > 0:
                detected.append(f"{name}:{count}")
        
        if total_count > best_count:
            best_count = total_count
            best_image = result_image
            best_config = config_name
    
    config_str = f"({best_config})" if best_config else ""
    print(f"   {best_count} sigs {config_str}", end="")
    
    return best_count, best_image


def update_json_file(json_path, updated_counts):
    """Update page signature counts in the JSON file."""
    with open(json_path, 'r') as file:
        data = json.load(file)

    data.update(updated_counts)

    with open(json_path, 'w') as file:
        json.dump(data, file, indent=2)

    print("\n✅ JSON updated")


# ==========================================================
# MAIN LOGIC (QUALITY-PRESERVED OPTIMIZATION)
# ==========================================================
def recheck_low_signature_pages(base_folder):
    """
    Optimized without compromising quality.
    Runs all strategies but with batch processing and reduced logging.
    Early stops at 3 signatures per page.
    """
    
    # Search for output.json
    possible_json_paths = [
        os.path.join(base_folder, "output.json"),
        os.path.join(base_folder, "signature_detected", "output.json"),
    ]
    
    json_path = None
    for path in possible_json_paths:
        if os.path.exists(path):
            json_path = path
            break
    
    if json_path is None:
        print("❌ output.json not found")
        return
    
    print(f"✅ output.json found")
    
    # Determine directories
    json_dir = os.path.dirname(json_path)
    parent_dir = os.path.dirname(json_dir) if os.path.basename(json_dir) == "signature_detected" else json_dir
    
    # Search for processed_images
    possible_processed_dirs = [
        os.path.join(parent_dir, "processed_images"),
        os.path.join(json_dir, "processed_images"),
    ]
    
    processed_dir = None
    for path in possible_processed_dirs:
        if os.path.exists(path):
            processed_dir = path
            break
    
    if processed_dir is None:
        print("❌ processed_images not found")
        return
    
    print(f"✅ processed_images found")

    # Load JSON
    with open(json_path, 'r') as file:
        page_counts = json.load(file)

    pages_to_recheck = [page for page, count in page_counts.items() if count < 2]

    if not pages_to_recheck:
        print("✅ All pages have ≥2 signatures")
        return

    print(f"\n🔍 Rechecking {len(pages_to_recheck)} pages")

    # Load model once
    print("📦 Loading model...", end=" ")
    processor = AutoImageProcessor.from_pretrained("tech4humans/conditional-detr-50-signature-detector")
    model = AutoModelForObjectDetection.from_pretrained("tech4humans/conditional-detr-50-signature-detector")
    model.eval()
    print("Done ✅")

    updated_counts = {}
    total_time = 0
    page_times = []

    for idx, page in enumerate(pages_to_recheck, 1):
        page_num = int(page.split("_")[1])
        image_path = os.path.join(processed_dir, f"processed_page_{page_num}.jpg")

        if not os.path.exists(image_path):
            continue

        print(f"\n[{idx}/{len(pages_to_recheck)}] {page} (curr:{page_counts[page]})")
        
        page_start = time.time()
        original_image = cv2.imread(image_path)
        best_count = page_counts[page]
        best_image = original_image.copy()
        best_method = "original"

        # Strategy 1: Split Detection
        print("  Split:", end=" ")
        count, merged = split_and_detect(processor, model, original_image.copy(), THRESHOLD, False)
        if count > best_count and count < 3:
            best_count, best_image, best_method = count, merged, "split"
        elif count >= 3:
            best_count, best_image, best_method = count, merged, "split"
            print(" ✅ EARLY STOP (3 sigs found)")
            elapsed = time.time() - page_start
            total_time += elapsed
            page_times.append((page, elapsed))
            
            if best_count > page_counts[page]:
                print(f"  ✅ {page_counts[page]}→{best_count} ({best_method}) [{elapsed:.1f}s]")
                updated_counts[page] = best_count
                # cv2.imwrite(image_path, best_image)
            continue
        print()

        # Strategy 2: Edge Detection (tests both 30% and 40%)
        print("  Edges:", end=" ")
        count, edges_img = detect_in_regions_batch(processor, model, original_image.copy(), THRESHOLD, False, 'edges', early_stop=True)
        if count > best_count and count < 3:
            best_count, best_image, best_method = count, edges_img, "edges"
        elif count >= 3:
            best_count, best_image, best_method = count, edges_img, "edges"
            print(" ✅ EARLY STOP (3 sigs found)")
            elapsed = time.time() - page_start
            total_time += elapsed
            page_times.append((page, elapsed))
            
            if best_count > page_counts[page]:
                print(f"  ✅ {page_counts[page]}→{best_count} ({best_method}) [{elapsed:.1f}s]")
                updated_counts[page] = best_count
                # cv2.imwrite(image_path, best_image)
            continue
        print()

        # Strategy 3: Corner Detection (tests both 40% and 50%)
        print("  Corners:", end=" ")
        count, corners_img = detect_in_regions_batch(processor, model, original_image.copy(), THRESHOLD, False, 'corners', early_stop=True)
        if count > best_count and count < 3:
            best_count, best_image, best_method = count, corners_img, "corners"
        elif count >= 3:
            best_count, best_image, best_method = count, corners_img, "corners"
            print(" ✅ EARLY STOP (3 sigs found)")
            elapsed = time.time() - page_start
            total_time += elapsed
            page_times.append((page, elapsed))
            
            if best_count > page_counts[page]:
                print(f"  ✅ {page_counts[page]}→{best_count} ({best_method}) [{elapsed:.1f}s]")
                updated_counts[page] = best_count
                # cv2.imwrite(image_path, best_image)
            continue
        print()

        # Strategy 4: Corners on Horizontal Flip
        print("  Corners-HFlip:", end=" ")
        h_flipped = mirror_horizontal(original_image)
        count, corners_img = detect_in_regions_batch(processor, model, h_flipped.copy(), THRESHOLD, False, 'corners', early_stop=True)
        if count > best_count and count < 3:
            best_count, best_image, best_method = count, corners_img, "corners_hflip"
        elif count >= 3:
            best_count, best_image, best_method = count, corners_img, "corners_hflip"
            print(" ✅ EARLY STOP (3 sigs found)")
        print()

        elapsed = time.time() - page_start
        total_time += elapsed
        page_times.append((page, elapsed))

        # Update if improved
        if best_count > page_counts[page]:
            print(f"  ✅ {page_counts[page]}→{best_count} ({best_method}) [{elapsed:.1f}s]")
            updated_counts[page] = best_count
            # cv2.imwrite(image_path, best_image)
        else:
            print(f"  ❌ No improvement [{elapsed:.1f}s]")

    # Update JSON
    if updated_counts:
        update_json_file(json_path, updated_counts)

    # Timing summary
    print("\n" + "="*50)
    print("⏱️  TIMING SUMMARY")
    print("="*50)
    if page_times:
        print(f"Total: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"Average: {total_time/len(page_times):.1f}s/page")
        print(f"Fastest: {min(page_times, key=lambda x: x[1])[1]:.1f}s")
        print(f"Slowest: {max(page_times, key=lambda x: x[1])[1]:.1f}s")

    print("\n🎯 Done!")


# ==========================================================
# RUN SCRIPT
# ==========================================================
if __name__ == "__main__":
    base_folder = r"C:\Users\admin\Desktop\HFS_Signature_Detection\SL - 694000"
    # base_folder = r"C:\Users\admin\Desktop\HFS_Signature_Detection\sanction latter english"
    
    recheck_low_signature_pages(base_folder)