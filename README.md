# Logo-Detection-With-OpenCv
Detects logos in PDF documents using OpenCV’s template matching and SIFT algorithms. Converts each PDF page to an image, scans for multiple logos, highlights matches, and saves detection results with a summary text file indicating true/false if all logos are found.

PDF Logo Detector

This Python script detects logos within PDF files using OpenCV. It supports multiple logo templates and uses template matching and SIFT feature detection for high accuracy.

🚀 Features

Detects multiple logos across all PDF pages

Supports .png, .jpg, .jpeg, .bmp logo formats

Multiscale and SIFT-based detection

Saves annotated output images

Generates a result.txt file with:

true → all logos found

false → some logos missing

🧰 Requirements

Install dependencies:

pip install opencv-python-headless numpy pdf2image


Also ensure Poppler is installed for PDF to image conversion.

🗂 Folder Structure
logo_detection/
│
├── Available_Logo/          # Folder containing logo images
├── LoanAgreement.pdf        # Target PDF file
├── output/                  # Generated output (detections + result.txt)
└── logo_detector.py         # Main detection script

⚙️ Usage

Edit file paths at the bottom of the script:

LOGO_FOLDER = r"C:\path\to\Available_Logo"
PDF_PATH = r"C:\path\to\document.pdf"
OUTPUT_FOLDER = r"C:\path\to\output"


Then run:

python logo_detector.py

🧾 Output

Annotated images saved as .png and .jpg

result.txt with detection status (true / false)

Console log showing detection summary per logo
