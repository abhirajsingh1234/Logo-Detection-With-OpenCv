# import os
# import cv2
# import numpy as np
# import json
# import torch
# import time
# import base64
# import warnings
# from pdf2image import convert_from_path
# from transformers import AutoImageProcessor, AutoModelForObjectDetection

# warnings.simplefilter("ignore")

# # ===================== CONFIG =====================
# THRESHOLD = 0.3

# # ===================== HELPER FUNCTIONS =====================

# def file_to_base64(file_path):
#     """Convert an image file to base64 string."""
#     try:
#         with open(file_path, "rb") as f:
#             return base64.b64encode(f.read()).decode("utf-8")
#     except Exception as e:
#         print(f"Error encoding {file_path}: {e}")
#         return None

# def clear_output_directory(folder):
#     """Delete all files in the output folder."""
#     if os.path.exists(folder):
#         for file in os.listdir(folder):
#             file_path = os.path.join(folder, file)
#             try:
#                 if os.path.isfile(file_path):
#                     os.unlink(file_path)
#             except Exception as e:
#                 print(f"Error deleting {file_path}: {e}")

# def convert_pdf_to_images(pdf_path):
#     """Convert PDF pages to PIL images."""
#     return convert_from_path(pdf_path)

# def get_document_name(file_path):
#     """Extract document name without extension from file path."""
#     base_name = os.path.basename(file_path)
#     return os.path.splitext(base_name)[0]

# # ===================== SIGNATURE DETECTION (PHASE 1) =====================

# def process_image_detr(processor, model, image, page_number=None, processed_dir=None, 
#                        signature_dir=None, threshold=0.3):
#     """Detect signatures in a single image using Conditional DETR."""
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     inputs = processor(images=image_rgb, return_tensors="pt")

#     with torch.no_grad():
#         outputs = model(**inputs)

#     target_sizes = torch.tensor([image_rgb.shape[:2]])
#     results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

#     signature_count = 0
#     cropped_images = []

#     for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
#         if int(label) == 0 and score > threshold:
#             signature_count += 1
#             x1, y1, x2, y2 = map(int, box.tolist())
#             cropped = image[y1:y2, x1:x2]

#             cropped_filename = f"signature_page_{page_number}_image_{i+1}.jpg"
#             cropped_path = os.path.join(signature_dir, cropped_filename)
#             cv2.imwrite(cropped_path, cropped)
#             cropped_images.append(cropped_path)

#             cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
#             cv2.putText(image, f"Signature {score:.2f}", (x1, y1-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

#     if processed_dir:
#         processed_path = os.path.join(processed_dir, f"processed_page_{page_number}.jpg")
#         cv2.imwrite(processed_path, image)

#     return signature_count, cropped_images

# def detect_signatures(input_path, base_output_dir=None):
#     """
#     Detect signatures in PDF or image files.
    
#     Args:
#         input_path: Path to the input PDF or image file
#         base_output_dir: Base directory for output. If None, uses current directory.
    
#     Returns:
#         Dictionary containing detection results and file paths
#     """
#     print("\n" + "="*60)
#     print("PHASE 1: INITIAL SIGNATURE DETECTION")
#     print("="*60)
    
#     doc_name = get_document_name(input_path)
    
#     if base_output_dir is None:
#         base_output_dir = os.path.dirname(input_path) or "."
    
#     doc_folder = os.path.join(base_output_dir, doc_name)
#     processed_dir = os.path.join(doc_folder, "processed_images")
#     signature_dir = os.path.join(doc_folder, "signature_detected")
    
#     os.makedirs(processed_dir, exist_ok=True)
#     os.makedirs(signature_dir, exist_ok=True)
    
#     clear_output_directory(processed_dir)
#     clear_output_directory(signature_dir)

#     print("Loading Conditional DETR signature detection model...")
#     processor = AutoImageProcessor.from_pretrained("tech4humans/conditional-detr-50-signature-detector")
#     model = AutoModelForObjectDetection.from_pretrained("tech4humans/conditional-detr-50-signature-detector")
#     model.eval()
#     print("✅ Model loaded successfully.")

#     signature_found = False
#     cropped_images_all = []
#     page_signature_count = {}

#     if input_path.lower().endswith(".pdf"):
#         images = convert_pdf_to_images(input_path)
#         print(f"Processing {len(images)} PDF pages...")
#         for page_number, img in enumerate(images, start=1):
#             image_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#             sig_count, cropped = process_image_detr(
#                 processor, model, image_cv, page_number, 
#                 processed_dir, signature_dir
#             )
#             page_signature_count[f"page_{page_number}"] = sig_count
#             if sig_count > 0:
#                 signature_found = True
#             cropped_images_all.extend(cropped)
#             print(f"  Page {page_number}: {sig_count} signature(s)")

#     elif input_path.lower().endswith((".jpg", ".jpeg", ".png")):
#         print("Processing image...")
#         image_cv = cv2.imread(input_path)
#         sig_count, cropped = process_image_detr(
#             processor, model, image_cv, page_number=1, 
#             processed_dir=processed_dir, signature_dir=signature_dir
#         )
#         if sig_count > 0:
#             signature_found = True
#             page_signature_count["page_1"] = sig_count
#         cropped_images_all.extend(cropped)
#         print(f"  Image: {sig_count} signature(s)")
#     else:
#         print("❌ Unsupported file type.")
#         return None

#     output_json_path = os.path.join(doc_folder, "output.json")
#     with open(output_json_path, 'w') as json_file:
#         json.dump(page_signature_count, json_file, indent=2)

#     print("\n=== PHASE 1 RESULT ===")
#     print(f"Document: {doc_name}")
#     print(f"Signatures detected: {'Yes' if signature_found else 'No'}")
#     print(f"Processed images: {os.path.abspath(processed_dir)}")
#     print(f"Cropped signatures: {os.path.abspath(signature_dir)}")
#     print(f"Output JSON: {os.path.abspath(output_json_path)}")

#     base64_images = [{"file_name": os.path.basename(p), "file_base64": file_to_base64(p)} 
#                      for p in cropped_images_all]

#     return {
#         "document_name": doc_name,
#         "signature_detected": "yes" if signature_found else "no",
#         "page_signature_count": page_signature_count,
#         "output_json_path": os.path.abspath(output_json_path),
#         "processed_images_dir": os.path.abspath(processed_dir),
#         "signature_detected_dir": os.path.abspath(signature_dir),
#         "cropped_images": cropped_images_all,
#         "base64_data": base64_images,
#         "model": (processor, model),
#         "doc_folder": doc_folder
#     }

# # ===================== RECHECK LOW SIGNATURES (PHASE 2) =====================

# def mirror_horizontal(image):
#     """Return a horizontally flipped image."""
#     return cv2.flip(image, 1)

# def process_image_detr_simple(processor, model, image, threshold=0.3):
#     """Detect signatures without saving (for optimization phase)."""
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     inputs = processor(images=image_rgb, return_tensors="pt")

#     with torch.no_grad():
#         outputs = model(**inputs)

#     target_sizes = torch.tensor([image_rgb.shape[:2]])
#     results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

#     sig_count = 0
#     for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#         if int(label) == 0 and score > threshold:
#             sig_count += 1

#     return sig_count

# def split_and_detect(processor, model, image, threshold=0.3):
#     """Split image into left and right halves, detect separately."""
#     h, w = image.shape[:2]
#     mid = w // 2
    
#     left_half = image[:, :mid].copy()
#     right_half = image[:, mid:].copy()
    
#     left_count = process_image_detr_simple(processor, model, left_half, threshold)
#     right_count = process_image_detr_simple(processor, model, right_half, threshold)
    
#     total_count = left_count + right_count
#     print(f"   L:{left_count} R:{right_count} = {total_count}", end="")
    
#     return total_count

# def detect_in_regions_batch(processor, model, image, threshold=0.3, region_type='edges', early_stop=False):
#     """Optimized batch detection in regions (edges or corners)."""
#     h, w = image.shape[:2]
    
#     if region_type == 'edges':
#         configs = [('30%', 0.3), ('40%', 0.4)]
#     else:
#         configs = [('40%', 0.4), ('50%', 0.5)]
    
#     best_count = 0
#     best_config = None
    
#     for config_name, size in configs:
#         if early_stop and best_count >= 3:
#             break
            
#         if region_type == 'edges':
#             edge_h = int(h * size)
#             edge_w = int(w * size)
            
#             regions = {
#                 't': image[0:edge_h, :].copy(),
#                 'b': image[h-edge_h:h, :].copy(),
#                 'l': image[:, 0:edge_w].copy(),
#                 'r': image[:, w-edge_w:w].copy()
#             }
#         else:
#             corner_h = int(h * size)
#             corner_w = int(w * size)
            
#             regions = {
#                 'TL': image[0:corner_h, 0:corner_w].copy(),
#                 'TR': image[0:corner_h, w-corner_w:w].copy(),
#                 'BL': image[h-corner_h:h, 0:corner_w].copy(),
#                 'BR': image[h-corner_h:h, w-corner_w:w].copy()
#             }
        
#         total_count = 0
        
#         for name, region_img in regions.items():
#             if early_stop and total_count >= 3:
#                 break
#             count = process_image_detr_simple(processor, model, region_img, threshold)
#             total_count += count
        
#         if total_count > best_count:
#             best_count = total_count
#             best_config = config_name
    
#     config_str = f"({best_config})" if best_config else ""
#     print(f"   {best_count} sigs {config_str}", end="")
    
#     return best_count

# def update_json_file(json_path, updated_counts):
#     """Update page signature counts in the JSON file."""
#     with open(json_path, 'r') as file:
#         data = json.load(file)

#     data.update(updated_counts)

#     with open(json_path, 'w') as file:
#         json.dump(data, file, indent=2)

#     print("✅ JSON updated")

# def recheck_low_signature_pages(processor, model, doc_folder, json_path, processed_dir):
#     """Recheck pages with fewer than 2 signatures using advanced strategies."""
    
#     print("\n" + "="*60)
#     print("PHASE 2: OPTIMIZED RECHECK FOR LOW SIGNATURE PAGES")
#     print("="*60)

#     with open(json_path, 'r') as file:
#         page_counts = json.load(file)

#     pages_to_recheck = [page for page, count in page_counts.items() if count < 2]

#     if not pages_to_recheck:
#         print("✅ All pages have ≥2 signatures. No recheck needed.")
#         return page_counts

#     print(f"🔍 Rechecking {len(pages_to_recheck)} pages with <2 signatures\n")

#     updated_counts = {}
#     total_time = 0
#     page_times = []

#     for idx, page in enumerate(pages_to_recheck, 1):
#         page_num = int(page.split("_")[1])
#         image_path = os.path.join(processed_dir, f"processed_page_{page_num}.jpg")

#         if not os.path.exists(image_path):
#             continue

#         print(f"[{idx}/{len(pages_to_recheck)}] {page} (current: {page_counts[page]})")
        
#         page_start = time.time()
#         original_image = cv2.imread(image_path)
#         best_count = page_counts[page]
#         best_method = "original"

#         # Strategy 1: Split Detection
#         print("  Split:", end=" ")
#         count = split_and_detect(processor, model, original_image.copy(), THRESHOLD)
#         if count > best_count:
#             best_count, best_method = count, "split"
#         if count >= 3:
#             print(" ✅ EARLY STOP")
#             elapsed = time.time() - page_start
#             total_time += elapsed
#             page_times.append((page, elapsed))
#             if best_count > page_counts[page]:
#                 print(f"  ✅ {page_counts[page]}→{best_count} ({best_method}) [{elapsed:.1f}s]")
#                 updated_counts[page] = best_count
#             continue
#         print()

#         # Strategy 2: Edge Detection
#         print("  Edges:", end=" ")
#         count = detect_in_regions_batch(processor, model, original_image.copy(), THRESHOLD, 'edges', early_stop=True)
#         if count > best_count:
#             best_count, best_method = count, "edges"
#         if count >= 3:
#             print(" ✅ EARLY STOP")
#             elapsed = time.time() - page_start
#             total_time += elapsed
#             page_times.append((page, elapsed))
#             if best_count > page_counts[page]:
#                 print(f"  ✅ {page_counts[page]}→{best_count} ({best_method}) [{elapsed:.1f}s]")
#                 updated_counts[page] = best_count
#             continue
#         print()

#         # Strategy 3: Corner Detection
#         print("  Corners:", end=" ")
#         count = detect_in_regions_batch(processor, model, original_image.copy(), THRESHOLD, 'corners', early_stop=True)
#         if count > best_count:
#             best_count, best_method = count, "corners"
#         if count >= 3:
#             print(" ✅ EARLY STOP")
#             elapsed = time.time() - page_start
#             total_time += elapsed
#             page_times.append((page, elapsed))
#             if best_count > page_counts[page]:
#                 print(f"  ✅ {page_counts[page]}→{best_count} ({best_method}) [{elapsed:.1f}s]")
#                 updated_counts[page] = best_count
#             continue
#         print()

#         # Strategy 4: Corners on Horizontal Flip
#         print("  Corners-HFlip:", end=" ")
#         h_flipped = mirror_horizontal(original_image)
#         count = detect_in_regions_batch(processor, model, h_flipped.copy(), THRESHOLD, 'corners', early_stop=True)
#         if count > best_count:
#             best_count, best_method = count, "corners_hflip"
#         print()

#         elapsed = time.time() - page_start
#         total_time += elapsed
#         page_times.append((page, elapsed))

#         if best_count > page_counts[page]:
#             print(f"  ✅ {page_counts[page]}→{best_count} ({best_method}) [{elapsed:.1f}s]")
#             updated_counts[page] = best_count
#         else:
#             print(f"  ❌ No improvement [{elapsed:.1f}s]")

#     if updated_counts:
#         update_json_file(json_path, updated_counts)
#         # Reload updated counts
#         with open(json_path, 'r') as file:
#             page_counts = json.load(file)

#     print("\n" + "="*60)
#     print("⏱️  TIMING SUMMARY")
#     print("="*60)
#     if page_times:
#         print(f"Total: {total_time:.1f}s ({total_time/60:.1f}min)")
#         print(f"Average: {total_time/len(page_times):.1f}s/page")
#         print(f"Fastest: {min(page_times, key=lambda x: x[1])[1]:.1f}s")
#         print(f"Slowest: {max(page_times, key=lambda x: x[1])[1]:.1f}s")

#     return page_counts

# # ===================== MAIN UNIFIED PIPELINE =====================

# def run_complete_detection_pipeline(input_path, base_output_dir=None):
#     """
#     Complete pipeline: Phase 1 (initial detection) + Phase 2 (optimization).
#     """
#     print("\n" + "="*60)
#     print("SIGNATURE DETECTION PIPELINE (Combined)")
#     print("="*60)

#     # PHASE 1: Initial Detection
#     phase1_result = detect_signatures(input_path, base_output_dir)
    
#     if phase1_result is None:
#         print("❌ Phase 1 failed. Exiting.")
#         return None

#     processor, model = phase1_result["model"]
#     json_path = phase1_result["output_json_path"]
#     processed_dir = phase1_result["processed_images_dir"]
#     doc_folder = phase1_result["doc_folder"]

#     # PHASE 2: Optimization for low signature pages
#     final_counts = recheck_low_signature_pages(processor, model, doc_folder, json_path, processed_dir)

#     # Final Summary
#     print("\n" + "="*60)
#     print("FINAL SUMMARY")
#     print("="*60)
#     print(f"Document: {phase1_result['document_name']}")
#     print(f"Total pages: {len(final_counts)}")
    
#     pages_with_sigs = sum(1 for count in final_counts.values() if count > 0)
#     print(f"Pages with signatures: {pages_with_sigs}")
    
#     for page, count in sorted(final_counts.items(), key=lambda x: int(x[0].split("_")[1])):
#         print(f"  {page}: {count} signature(s)")

#     print(f"\nOutput directory: {os.path.abspath(doc_folder)}")
#     print("✅ Pipeline complete!")

#     return {
#         "phase1_result": phase1_result,
#         "final_counts": final_counts
#     }

# # ===================== RUN SCRIPT =====================

# if __name__ == "__main__":
#     input_path = r"C:\Users\admin\Desktop\HFS_Signature_Detection\SL - 694000.pdf"
#     # input_path = r"C:\Users\admin\Desktop\HFS_Signature_Detection\DRF 1.pdf"
#     # input_path = r"C:\Users\admin\Desktop\HFS_Signature_Detection\sanction latter english.pdf"
    
#     result = run_complete_detection_pipeline(input_path)



import os
import cv2
import numpy as np
import json
import torch
import time
import base64
import warnings
from pdf2image import convert_from_path
from transformers import AutoImageProcessor, AutoModelForObjectDetection

warnings.simplefilter("ignore")

# ===================== CONFIG =====================
THRESHOLD = 0.3

# ===================== HELPER FUNCTIONS =====================

def file_to_base64(file_path):
    """Convert an image file to base64 string."""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding {file_path}: {e}")
        return None

def clear_output_directory(folder):
    """Delete all files in the output folder."""
    if os.path.exists(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def convert_pdf_to_images(pdf_path):
    """Convert PDF pages to PIL images."""
    return convert_from_path(pdf_path)

def get_document_name(file_path):
    """Extract document name without extension from file path."""
    base_name = os.path.basename(file_path)
    return os.path.splitext(base_name)[0]

# ===================== SIGNATURE DETECTION (PHASE 1) =====================

def process_image_detr(processor, model, image, page_number=None, threshold=0.3):
    """Detect signatures in a single image using Conditional DETR."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image_rgb.shape[:2]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    signature_count = 0
    # cropped_images = []  # Commented out - no longer needed

    for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
        if int(label) == 0 and score > threshold:
            signature_count += 1
            # x1, y1, x2, y2 = map(int, box.tolist())
            # cropped = image[y1:y2, x1:x2]

            # cropped_filename = f"signature_page_{page_number}_image_{i+1}.jpg"
            # cropped_path = os.path.join(signature_dir, cropped_filename)
            # cv2.imwrite(cropped_path, cropped)
            # cropped_images.append(cropped_path)

            # cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
            # cv2.putText(image, f"Signature {score:.2f}", (x1, y1-10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # if processed_dir:
    #     processed_path = os.path.join(processed_dir, f"processed_page_{page_number}.jpg")
    #     cv2.imwrite(processed_path, image)

    return signature_count  # , cropped_images  # Removed cropped_images return

def detect_signatures(input_path, base_output_dir=None):
    """
    Detect signatures in PDF or image files.
    
    Args:
        input_path: Path to the input PDF or image file
        base_output_dir: Base directory for output. If None, uses current directory.
    
    Returns:
        Dictionary containing detection results and file paths
    """
    print("\n" + "="*60)
    print("PHASE 1: INITIAL SIGNATURE DETECTION")
    print("="*60)
    
    doc_name = get_document_name(input_path)
    
    if base_output_dir is None:
        base_output_dir = os.path.dirname(input_path) or "."
    
    doc_folder = os.path.join(base_output_dir, doc_name)
    # processed_dir = os.path.join(doc_folder, "processed_images")
    # signature_dir = os.path.join(doc_folder, "signature_detected")
    
    os.makedirs(doc_folder, exist_ok=True)
    # os.makedirs(processed_dir, exist_ok=True)
    # os.makedirs(signature_dir, exist_ok=True)
    
    # clear_output_directory(processed_dir)
    # clear_output_directory(signature_dir)

    print("Loading Conditional DETR signature detection model...")
    processor = AutoImageProcessor.from_pretrained("tech4humans/conditional-detr-50-signature-detector")
    model = AutoModelForObjectDetection.from_pretrained("tech4humans/conditional-detr-50-signature-detector")
    model.eval()
    print("✅ Model loaded successfully.")

    signature_found = False
    # cropped_images_all = []
    page_signature_count = {}

    if input_path.lower().endswith(".pdf"):
        images = convert_pdf_to_images(input_path)
        print(f"Processing {len(images)} PDF pages...")
        for page_number, img in enumerate(images, start=1):
            image_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            sig_count = process_image_detr(
                processor, model, image_cv, page_number
            )
            page_signature_count[f"page_{page_number}"] = sig_count
            if sig_count > 0:
                signature_found = True
            # cropped_images_all.extend(cropped)
            print(f"  Page {page_number}: {sig_count} signature(s)")

    elif input_path.lower().endswith((".jpg", ".jpeg", ".png")):
        print("Processing image...")
        image_cv = cv2.imread(input_path)
        sig_count = process_image_detr(
            processor, model, image_cv, page_number=1
        )
        if sig_count > 0:
            signature_found = True
            page_signature_count["page_1"] = sig_count
        # cropped_images_all.extend(cropped)
        print(f"  Image: {sig_count} signature(s)")
    else:
        print("❌ Unsupported file type.")
        return None

    output_json_path = os.path.join(doc_folder, "output.json")
    # Save in final format from the start
    detected = all(count >= 2 for count in page_signature_count.values()) if page_signature_count else False
    pages_with_less_than_2 = [page for page, count in page_signature_count.items() if count < 2]
    
    initial_json_data = {
        "detected": detected,
        "document_name": doc_name,
        "page_signature_count": page_signature_count,
        "pages_with_less_than_2_signatures": pages_with_less_than_2
    }
    
    with open(output_json_path, 'w') as json_file:
        json.dump(initial_json_data, json_file, indent=2)

    print("\n=== PHASE 1 RESULT ===")
    print(f"Document: {doc_name}")
    print(f"Signatures detected: {'Yes' if signature_found else 'No'}")
    # print(f"Processed images: {os.path.abspath(processed_dir)}")
    # print(f"Cropped signatures: {os.path.abspath(signature_dir)}")
    print(f"Output JSON: {os.path.abspath(output_json_path)}")

    # base64_images = [{"file_name": os.path.basename(p), "file_base64": file_to_base64(p)} 
    #                  for p in cropped_images_all]

    return {
        "document_name": doc_name,
        "signature_detected": "yes" if signature_found else "no",
        "page_signature_count": page_signature_count,
        "output_json_path": os.path.abspath(output_json_path),
        # "processed_images_dir": os.path.abspath(processed_dir),
        # "signature_detected_dir": os.path.abspath(signature_dir),
        # "cropped_images": cropped_images_all,
        # "base64_data": base64_images,
        "model": (processor, model),
        "doc_folder": doc_folder
    }

# ===================== RECHECK LOW SIGNATURES (PHASE 2) =====================

def mirror_horizontal(image):
    """Return a horizontally flipped image."""
    return cv2.flip(image, 1)

def process_image_detr_simple(processor, model, image, threshold=0.3):
    """Detect signatures without saving (for optimization phase)."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image_rgb.shape[:2]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    sig_count = 0
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if int(label) == 0 and score > threshold:
            sig_count += 1

    return sig_count

def split_and_detect(processor, model, image, threshold=0.3):
    """Split image into left and right halves, detect separately."""
    h, w = image.shape[:2]
    mid = w // 2
    
    left_half = image[:, :mid].copy()
    right_half = image[:, mid:].copy()
    
    left_count = process_image_detr_simple(processor, model, left_half, threshold)
    right_count = process_image_detr_simple(processor, model, right_half, threshold)
    
    total_count = left_count + right_count
    print(f"   L:{left_count} R:{right_count} = {total_count}", end="")
    
    return total_count

def detect_in_regions_batch(processor, model, image, threshold=0.3, region_type='edges', early_stop=False):
    """Optimized batch detection in regions (edges or corners)."""
    h, w = image.shape[:2]
    
    if region_type == 'edges':
        configs = [('30%', 0.3), ('40%', 0.4)]
    else:
        configs = [('40%', 0.4), ('50%', 0.5)]
    
    best_count = 0
    best_config = None
    
    for config_name, size in configs:
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
        else:
            corner_h = int(h * size)
            corner_w = int(w * size)
            
            regions = {
                'TL': image[0:corner_h, 0:corner_w].copy(),
                'TR': image[0:corner_h, w-corner_w:w].copy(),
                'BL': image[h-corner_h:h, 0:corner_w].copy(),
                'BR': image[h-corner_h:h, w-corner_w:w].copy()
            }
        
        total_count = 0
        
        for name, region_img in regions.items():
            if early_stop and total_count >= 3:
                break
            count = process_image_detr_simple(processor, model, region_img, threshold)
            total_count += count
        
        if total_count > best_count:
            best_count = total_count
            best_config = config_name
    
    config_str = f"({best_config})" if best_config else ""
    print(f"   {best_count} sigs {config_str}", end="")
    
    return best_count

def update_json_file(json_path, updated_counts):
    """Update page signature counts in the JSON file."""
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Update page counts
    data["page_signature_count"].update(updated_counts)
    
    # Recalculate detected flag
    data["detected"] = all(count >= 2 for count in data["page_signature_count"].values())
    
    # Recalculate pages with less than 2 signatures
    data["pages_with_less_than_2_signatures"] = [
        page for page, count in data["page_signature_count"].items() if count < 2
    ]

    with open(json_path, 'w') as file:
        json.dump(data, file, indent=2)

    print("✅ JSON updated")


def recheck_low_signature_pages(processor, model, doc_folder, json_path, input_path):
    """Recheck pages with fewer than 2 signatures using advanced strategies."""
    
    print("\n" + "="*60)
    print("PHASE 2: OPTIMIZED RECHECK FOR LOW SIGNATURE PAGES")
    print("="*60)

    with open(json_path, 'r') as file:
        json_data = json.load(file)
    
    # Extract page_signature_count from the new JSON format
    page_counts = json_data.get("page_signature_count", {})

    pages_to_recheck = [page for page, count in page_counts.items() if count < 2]

    if not pages_to_recheck:
        print("✅ All pages have ≥2 signatures. No recheck needed.")
        return page_counts

    print(f"🔍 Rechecking {len(pages_to_recheck)} pages with <2 signatures\n")

    # Load images based on file type
    if input_path.lower().endswith(".pdf"):
        images = convert_pdf_to_images(input_path)
    elif input_path.lower().endswith((".jpg", ".jpeg", ".png")):
        images = [cv2.imread(input_path)]
    else:
        print("❌ Unsupported file type.")
        return page_counts

    updated_counts = {}
    total_time = 0
    page_times = []

    for idx, page in enumerate(pages_to_recheck, 1):
        page_num = int(page.split("_")[1])
        
        # Get image from loaded images
        if input_path.lower().endswith(".pdf"):
            if page_num <= len(images):
                original_image = cv2.cvtColor(np.array(images[page_num - 1]), cv2.COLOR_RGB2BGR)
            else:
                continue
        else:
            original_image = images[0]

        print(f"[{idx}/{len(pages_to_recheck)}] {page} (current: {page_counts[page]})")
        
        page_start = time.time()
        best_count = page_counts[page]
        best_method = "original"

        # Strategy 1: Split Detection
        print("  Split:", end=" ")
        count = split_and_detect(processor, model, original_image.copy(), THRESHOLD)
        if count > best_count:
            best_count, best_method = count, "split"
        if count >= 3:
            print(" ✅ EARLY STOP")
            elapsed = time.time() - page_start
            total_time += elapsed
            page_times.append((page, elapsed))
            if best_count > page_counts[page]:
                print(f"  ✅ {page_counts[page]}→{best_count} ({best_method}) [{elapsed:.1f}s]")
                updated_counts[page] = best_count
            continue
        print()

        # Strategy 2: Edge Detection
        print("  Edges:", end=" ")
        count = detect_in_regions_batch(processor, model, original_image.copy(), THRESHOLD, 'edges', early_stop=True)
        if count > best_count:
            best_count, best_method = count, "edges"
        if count >= 3:
            print(" ✅ EARLY STOP")
            elapsed = time.time() - page_start
            total_time += elapsed
            page_times.append((page, elapsed))
            if best_count > page_counts[page]:
                print(f"  ✅ {page_counts[page]}→{best_count} ({best_method}) [{elapsed:.1f}s]")
                updated_counts[page] = best_count
            continue
        print()

        # Strategy 3: Corner Detection
        print("  Corners:", end=" ")
        count = detect_in_regions_batch(processor, model, original_image.copy(), THRESHOLD, 'corners', early_stop=True)
        if count > best_count:
            best_count, best_method = count, "corners"
        if count >= 3:
            print(" ✅ EARLY STOP")
            elapsed = time.time() - page_start
            total_time += elapsed
            page_times.append((page, elapsed))
            if best_count > page_counts[page]:
                print(f"  ✅ {page_counts[page]}→{best_count} ({best_method}) [{elapsed:.1f}s]")
                updated_counts[page] = best_count
            continue
        print()

        # Strategy 4: Corners on Horizontal Flip
        print("  Corners-HFlip:", end=" ")
        h_flipped = mirror_horizontal(original_image)
        count = detect_in_regions_batch(processor, model, h_flipped.copy(), THRESHOLD, 'corners', early_stop=True)
        if count > best_count:
            best_count, best_method = count, "corners_hflip"
        print()

        elapsed = time.time() - page_start
        total_time += elapsed
        page_times.append((page, elapsed))

        if best_count > page_counts[page]:
            print(f"  ✅ {page_counts[page]}→{best_count} ({best_method}) [{elapsed:.1f}s]")
            updated_counts[page] = best_count
        else:
            print(f"  ❌ No improvement [{elapsed:.1f}s]")

    if updated_counts:
        update_json_file(json_path, updated_counts)
        # Reload updated counts
        with open(json_path, 'r') as file:
            json_data = json.load(file)
            page_counts = json_data.get("page_signature_count", {})

    print("\n" + "="*60)
    print("⏱️  TIMING SUMMARY")
    print("="*60)
    if page_times:
        print(f"Total: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"Average: {total_time/len(page_times):.1f}s/page")
        print(f"Fastest: {min(page_times, key=lambda x: x[1])[1]:.1f}s")
        print(f"Slowest: {max(page_times, key=lambda x: x[1])[1]:.1f}s")

    return page_counts

# ===================== MAIN UNIFIED PIPELINE =====================

def run_complete_detection_pipeline(input_path, base_output_dir=None):
    """
    Complete pipeline: Phase 1 (initial detection) + Phase 2 (optimization).
    """
    print("\n" + "="*60)
    print("SIGNATURE DETECTION PIPELINE (Combined)")
    print("="*60)

    # PHASE 1: Initial Detection
    phase1_result = detect_signatures(input_path, base_output_dir)
    
    if phase1_result is None:
        print("❌ Phase 1 failed. Exiting.")
        return None

    processor, model = phase1_result["model"]
    json_path = phase1_result["output_json_path"]
    doc_folder = phase1_result["doc_folder"]

    # PHASE 2: Optimization for low signature pages
    recheck_low_signature_pages(processor, model, doc_folder, json_path, input_path)

    # Load final JSON
    final_json_path = os.path.join(doc_folder, "output.json")
    with open(final_json_path, 'r') as f:
        final_json_data = json.load(f)
    
    final_counts = final_json_data.get("page_signature_count", {})

    # Final Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Document: {phase1_result['document_name']}")
    print(f"Total pages: {len(final_counts)}")
    
    pages_with_sigs = sum(1 for count in final_counts.values() if count > 0)
    print(f"Pages with signatures: {pages_with_sigs}")
    
    for page, count in sorted(final_counts.items(), key=lambda x: int(x[0].split("_")[1])):
        print(f"  {page}: {count} signature(s)")
    
    print(f"\nOutput directory: {os.path.abspath(doc_folder)}")
    print("✅ Pipeline complete!")

    return {
        "phase1_result": phase1_result,
        "final_counts": final_counts,
        "final_json": final_json_data
    }

# ===================== RUN SCRIPT =====================

if __name__ == "__main__":
    input_path = r"C:\Users\admin\Desktop\HFS_Signature_Detection\SL - 694000.pdf"
    input_path = r"C:\Users\admin\Desktop\HFS_Signature_Detection\SANCTION LETTER 1.pdf"
    # input_path = r"C:\Users\admin\Desktop\HFS_Signature_Detection\sanction latter english.pdf"
    
    result = run_complete_detection_pipeline(input_path)