

# import os
# import cv2
# import numpy as np
# from pdf2image import convert_from_path
# from transformers import AutoImageProcessor, AutoModelForObjectDetection
# import torch
# import base64
# import warnings
# import json

# warnings.simplefilter("ignore")

# # ===================== Helper Functions =====================

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

# # ===================== Signature Detection =====================

# def process_image_detr(processor, model, image, page_number=None, processed_dir=None, 
#                        signature_dir=None, threshold=0.3):
#     """Detect signatures in a single image using Conditional DETR."""
#     # Convert OpenCV BGR image to RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Preprocess
#     inputs = processor(images=image_rgb, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Get predicted boxes and scores
#     target_sizes = torch.tensor([image_rgb.shape[:2]])  # height, width
#     results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

#     signature_detected = False
#     cropped_images = []

#     for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
#         # Assuming signature class is label 0
#         if int(label) == 0 and score > threshold:
#             signature_detected = True
#             x1, y1, x2, y2 = map(int, box.tolist())
#             cropped = image[y1:y2, x1:x2]

#             # Save cropped signature image in signature_detected folder
#             cropped_filename = f"signature_page_{page_number}_image_{i+1}.jpg"
#             cropped_path = os.path.join(signature_dir, cropped_filename)
#             cv2.imwrite(cropped_path, cropped)
#             cropped_images.append(cropped_path)

#             # Draw bounding box on original image
#             cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
#             cv2.putText(image, f"Signature {score:.2f}", (x1, y1-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

#     # Save processed image in processed_images folder
#     if processed_dir:
#         processed_path = os.path.join(processed_dir, f"processed_page_{page_number}.jpg")
#         cv2.imwrite(processed_path, image)

#     return signature_detected, cropped_images

# # ===================== Main Function =====================

# def detect_signatures(input_path, base_output_dir=None):
#     """
#     Detect signatures in PDF or image files.
    
#     Args:
#         input_path: Path to the input PDF or image file
#         base_output_dir: Base directory for output. If None, uses current directory.
    
#     Returns:
#         Dictionary containing detection results and file paths
#     """
#     # Get document name
#     doc_name = get_document_name(input_path)
    
#     # Set base output directory
#     if base_output_dir is None:
#         base_output_dir = os.path.dirname(input_path) or "."
    
#     # Create organized folder structure
#     doc_folder = os.path.join(base_output_dir, doc_name)
#     processed_dir = os.path.join(doc_folder, "processed_images")
#     signature_dir = os.path.join(doc_folder, "signature_detected")
    
#     # Create directories
#     os.makedirs(processed_dir, exist_ok=True)
#     os.makedirs(signature_dir, exist_ok=True)
    
#     # Clear existing files
#     clear_output_directory(processed_dir)
#     clear_output_directory(signature_dir)

#     # Load Conditional DETR signature model
#     print("Loading Conditional DETR signature detection model...")
#     processor = AutoImageProcessor.from_pretrained("tech4humans/conditional-detr-50-signature-detector")
#     model = AutoModelForObjectDetection.from_pretrained("tech4humans/conditional-detr-50-signature-detector")
#     model.eval()
#     print("Model loaded successfully.")

#     signature_found = False
#     cropped_images_all = []

#     # Handle PDFs
#     if input_path.lower().endswith(".pdf"):
#         images = convert_pdf_to_images(input_path)
#         for page_number, img in enumerate(images, start=1):
#             image_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#             detected, cropped = process_image_detr(
#                 processor, model, image_cv, page_number, 
#                 processed_dir, signature_dir
#             )
#             if detected:
#                 signature_found = True
#             cropped_images_all.extend(cropped)

#     # Handle images
#     elif input_path.lower().endswith((".jpg", ".jpeg", ".png")):
#         image_cv = cv2.imread(input_path)
#         detected, cropped = process_image_detr(
#             processor, model, image_cv, page_number=1, 
#             processed_dir=processed_dir, signature_dir=signature_dir
#         )
#         if detected:
#             signature_found = True
#         cropped_images_all.extend(cropped)
#     else:
#         print("Unsupported file type.")
#         return

#     print("\n=== Detection Result ===")
#     print("Document name:", doc_name)
#     print("Signatures detected:", "Yes" if signature_found else "No")
#     print("Processed images saved in:", os.path.abspath(processed_dir))
#     print("Cropped signatures saved in:", os.path.abspath(signature_dir))

#     # Optional: Convert cropped images to base64
#     base64_images = [{"file_name": os.path.basename(p), "file_base64": file_to_base64(p)} 
#                      for p in cropped_images_all]

#     return {
#         "document_name": doc_name,
#         "signature_detected": "yes" if signature_found else "no",
#         "processed_images_dir": os.path.abspath(processed_dir),
#         "signature_detected_dir": os.path.abspath(signature_dir),
#         "cropped_images": cropped_images_all,
#         "base64_data": base64_images
#     }

# # ===================== Run Script =====================

# if __name__ == "__main__":
#     path = r"C:\Users\admin\Desktop\HFS_Signature_Detection\SANCTION LETTER 1.pdf"
#     # path = r"C:\Users\admin\Desktop\HFS_Signature_Detection\DRF 1.pdf"
#     result = detect_signatures(path)
#     print(json.dumps(result, indent=2, default=str))


import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch
import base64
import warnings
import json

warnings.simplefilter("ignore")

# ===================== Helper Functions =====================

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

# ===================== Signature Detection =====================

def process_image_detr(processor, model, image, page_number=None, processed_dir=None, 
                       signature_dir=None, threshold=0.3):
    """Detect signatures in a single image using Conditional DETR."""
    # Convert OpenCV BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess
    inputs = processor(images=image_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted boxes and scores
    target_sizes = torch.tensor([image_rgb.shape[:2]])  # height, width
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    signature_count = 0
    cropped_images = []

    for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
        # Assuming signature class is label 0
        if int(label) == 0 and score > threshold:
            signature_count += 1
            x1, y1, x2, y2 = map(int, box.tolist())
            cropped = image[y1:y2, x1:x2]

            # Save cropped signature image in signature_detected folder
            cropped_filename = f"signature_page_{page_number}_image_{i+1}.jpg"
            cropped_path = os.path.join(signature_dir, cropped_filename)
            cv2.imwrite(cropped_path, cropped)
            cropped_images.append(cropped_path)

            # Draw bounding box on original image
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(image, f"Signature {score:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Save processed image in processed_images folder
    if processed_dir:
        processed_path = os.path.join(processed_dir, f"processed_page_{page_number}.jpg")
        cv2.imwrite(processed_path, image)

    return signature_count, cropped_images

# ===================== Main Function =====================

def detect_signatures(input_path, base_output_dir=None):
    """
    Detect signatures in PDF or image files.
    
    Args:
        input_path: Path to the input PDF or image file
        base_output_dir: Base directory for output. If None, uses current directory.
    
    Returns:
        Dictionary containing detection results and file paths
    """
    # Get document name
    doc_name = get_document_name(input_path)
    
    # Set base output directory
    if base_output_dir is None:
        base_output_dir = os.path.dirname(input_path) or "."
    
    # Create organized folder structure
    doc_folder = os.path.join(base_output_dir, doc_name)
    processed_dir = os.path.join(doc_folder, "processed_images")
    signature_dir = os.path.join(doc_folder, "signature_detected")
    
    # Create directories
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(signature_dir, exist_ok=True)
    
    # Clear existing files
    clear_output_directory(processed_dir)
    clear_output_directory(signature_dir)

    # Load Conditional DETR signature model
    print("Loading Conditional DETR signature detection model...")
    processor = AutoImageProcessor.from_pretrained("tech4humans/conditional-detr-50-signature-detector")
    model = AutoModelForObjectDetection.from_pretrained("tech4humans/conditional-detr-50-signature-detector")
    model.eval()
    print("Model loaded successfully.")

    signature_found = False
    cropped_images_all = []
    page_signature_count = {}

    # Handle PDFs
    if input_path.lower().endswith(".pdf"):
        images = convert_pdf_to_images(input_path)
        for page_number, img in enumerate(images, start=1):
            image_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            sig_count, cropped = process_image_detr(
                processor, model, image_cv, page_number, 
                processed_dir, signature_dir
            )
            if sig_count > 0:
                signature_found = True
                page_signature_count[f"page_{page_number}"] = sig_count
            cropped_images_all.extend(cropped)

    # Handle images
    elif input_path.lower().endswith((".jpg", ".jpeg", ".png")):
        image_cv = cv2.imread(input_path)
        sig_count, cropped = process_image_detr(
            processor, model, image_cv, page_number=1, 
            processed_dir=processed_dir, signature_dir=signature_dir
        )
        if sig_count > 0:
            signature_found = True
            page_signature_count["page_1"] = sig_count
        cropped_images_all.extend(cropped)
    else:
        print("Unsupported file type.")
        return

    # Save page signature count to JSON file
    output_json_path = os.path.join(doc_folder, "output.json")
    with open(output_json_path, 'w') as json_file:
        json.dump(page_signature_count, json_file, indent=2)

    print("\n=== Detection Result ===")
    print("Document name:", doc_name)
    print("Signatures detected:", "Yes" if signature_found else "No")
    print("Processed images saved in:", os.path.abspath(processed_dir))
    print("Cropped signatures saved in:", os.path.abspath(signature_dir))
    print("Output JSON saved at:", os.path.abspath(output_json_path))

    # Optional: Convert cropped images to base64
    base64_images = [{"file_name": os.path.basename(p), "file_base64": file_to_base64(p)} 
                     for p in cropped_images_all]

    return {
        "document_name": doc_name,
        "signature_detected": "yes" if signature_found else "no",
        "page_signature_count": page_signature_count,
        "output_json_path": os.path.abspath(output_json_path),
        "processed_images_dir": os.path.abspath(processed_dir),
        "signature_detected_dir": os.path.abspath(signature_dir),
        "cropped_images": cropped_images_all,
        "base64_data": base64_images
    }

# ===================== Run Script =====================

if __name__ == "__main__":
    path = r"C:\Users\admin\Desktop\HFS_Signature_Detection\SANCTION LETTER 1.pdf"
    path = r"C:\Users\admin\Desktop\HFS_Signature_Detection\DRF 1.pdf"
    result = detect_signatures(path)
    print(json.dumps(result, indent=2, default=str))