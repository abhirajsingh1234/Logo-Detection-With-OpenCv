import cv2
import numpy as np
from pdf2image import convert_from_path
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

class LogoDetector:
    def __init__(self, logo_folder, pdf_path, output_folder='output'):
        self.logo_folder = Path(logo_folder)
        self.pdf_path = pdf_path
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.logos = self.load_logos()
        
    def load_logos(self):
        logos = {}
        supported_formats = ['.png', '.jpg', '.jpeg', '.bmp']
        for file in self.logo_folder.iterdir():
            if file.suffix.lower() in supported_formats:
                img = cv2.imread(str(file))
                if img is not None:
                    logos[file.stem] = img
                    print(f"Loaded logo: {file.name} (size: {img.shape[1]}x{img.shape[0]})")
        print(f"\nTotal logos loaded: {len(logos)}")
        return logos
    
    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray
    
    def match_template_multiscale(self, page_img, template, threshold=0.6):
        page_gray = self.preprocess_image(page_img)
        template_gray = self.preprocess_image(template)
        h, w = template_gray.shape
        best_match = None
        best_val = threshold
        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
        for scale in scales:
            new_w = int(w * scale)
            new_h = int(h * scale)
            if new_h > page_gray.shape[0] or new_w > page_gray.shape[1]:
                continue
            if new_h < 20 or new_w < 20:
                continue
            resized_template = cv2.resize(template_gray, (new_w, new_h))
            result = cv2.matchTemplate(page_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val > best_val:
                best_val = max_val
                best_match = {
                    'location': max_loc,
                    'width': new_w,
                    'height': new_h,
                    'confidence': max_val,
                    'scale': scale
                }
        return best_match
    
    def match_template_sift(self, page_img, template, threshold=0.6):
        try:
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(self.preprocess_image(template), None)
            kp2, des2 = sift.detectAndCompute(self.preprocess_image(page_img), None)
            if des1 is None or des2 is None or len(kp1) < 4:
                return None
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            if len(good_matches) >= 4:
                confidence = len(good_matches) / len(kp1)
                if confidence > threshold:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    x_min = int(np.min(dst_pts[:, 0, 0]))
                    y_min = int(np.min(dst_pts[:, 0, 1]))
                    x_max = int(np.max(dst_pts[:, 0, 0]))
                    y_max = int(np.max(dst_pts[:, 0, 1]))
                    return {
                        'location': (x_min, y_min),
                        'width': x_max - x_min,
                        'height': y_max - y_min,
                        'confidence': min(confidence, 1.0),
                        'matches': len(good_matches)
                    }
        except Exception as e:
            print(f"    SIFT matching error: {str(e)}")
        return None
    
    def draw_detection(self, img, match, logo_name):
        x, y = match['location']
        w, h = match['width'], match['height']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        label = f"{logo_name} ({match['confidence']:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img, (x, y - text_h - 15), (x + text_w + 10, y), (0, 255, 0), -1)
        cv2.putText(img, label, (x + 5, y - 8), font, font_scale, (0, 0, 0), thickness)
        return img
    
    def detect_logos_in_pdf(self, threshold=0.55, dpi=200, use_sift=True):
        print(f"\nConverting PDF to images (DPI={dpi})...")
        pages = convert_from_path(self.pdf_path, dpi=dpi)
        print(f"Total pages: {len(pages)}")
        
        found_logos = set()
        results = []
        
        for page_num, page in enumerate(pages, 1):
            print(f"\n{'='*60}")
            print(f"Processing page {page_num}/{len(pages)}...")
            print('='*60)
            page_cv = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            page_with_detections = page_cv.copy()
            page_has_detection = False
            
            for logo_name, logo_template in self.logos.items():
                if logo_name in found_logos:
                    continue  # ✅ Skip already found logos
                
                print(f"  🔍 Searching for: {logo_name}...")
                start_time = time.time()
                match = self.match_template_multiscale(page_cv, logo_template, threshold)
                if not match and use_sift:
                    print(f"    Trying SIFT feature matching...")
                    match = self.match_template_sift(page_cv, logo_template, threshold=0.3)
                elapsed = time.time() - start_time
                
                if match:
                    print(f"  ✅ FOUND {logo_name}!")
                    print(f"     Confidence: {match['confidence']:.3f}")
                    print(f"     Location: {match['location']}")
                    print(f"     Time: {elapsed:.2f}s")
                    page_with_detections = self.draw_detection(page_with_detections, match, logo_name)
                    found_logos.add(logo_name)  # ✅ Stop searching this logo in future pages
                    page_has_detection = True
                    results.append({
                        'logo': logo_name,
                        'page': page_num,
                        'confidence': match['confidence'],
                        'location': match['location']
                    })
                else:
                    print(f"  ❌ Not found ({elapsed:.2f}s)")
            
            if page_has_detection:
                output_path = self.output_folder / f'page_{page_num}_detections.png'
                cv2.imwrite(str(output_path), page_with_detections)
                print(f"\n  💾 Saved: {output_path}")
                output_path_hq = self.output_folder / f'page_{page_num}_detections_HQ.jpg'
                cv2.imwrite(str(output_path_hq), page_with_detections, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # ✅ Stop if all logos are found
            if len(found_logos) == len(self.logos):
                print(f"\n🎉 All logos found! Stopping search.")
                break
        
        # ✅ Write result.txt (true if all logos found, else false)
        result_file = self.output_folder / "result.txt"
        with open(result_file, "w") as f:
            if len(found_logos) == len(self.logos):
                f.write("true")
            else:
                f.write("false")
        print(f"\n📝 Result written to: {result_file}")
        
        print("\n" + "="*60)
        print("📊 DETECTION SUMMARY")
        print("="*60)
        if results:
            for r in results:
                print(f"\n✅ Logo: {r['logo']}")
                print(f"   Page: {r['page']}")
                print(f"   Confidence: {r['confidence']:.3f}")
                print(f"   Location: {r['location']}")
        else:
            print("\n❌ No logos detected.")
        
        missing = set(self.logos.keys()) - found_logos
        if missing:
            print(f"\n⚠️  Logos NOT found ({len(missing)}):")
            for logo in missing:
                print(f"   - {logo}")
        print(f"\n📁 Output folder: {self.output_folder.absolute()}")
        return results


if __name__ == "__main__":
    LOGO_FOLDER = r"C:\Users\admin\Desktop\logo_detection\Available_Logo"
    PDF_PATH = r"C:\Users\admin\Desktop\logo_detection\LoanAgreement-NAVI.pdf"
    PDF_PATH = r"C:\Users\admin\Desktop\logo_detection\LoanAgreement & Sanction letter -PTT.pdf"
    PDF_PATH = r"C:\Users\admin\Desktop\logo_detection\LoanAgreement-NAVI.pdf"
    OUTPUT_FOLDER = r"C:\Users\admin\Desktop\logo_detection\output"
    THRESHOLD = 0.75
    DPI = 200
    
    detector = LogoDetector(LOGO_FOLDER, PDF_PATH, OUTPUT_FOLDER)
    results = detector.detect_logos_in_pdf(threshold=THRESHOLD, dpi=DPI, use_sift=True)
    print("\n✅ Detection complete!")
