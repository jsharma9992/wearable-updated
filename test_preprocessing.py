import cv2
import argparse
from pathlib import Path
from intelligence.ocr_engine import OCREngine
import config

def test_preprocessing(image_path):
    engine = OCREngine() # No readers needed for preprocessing
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Could not load image: {image_path}")
        return
        
    config.OCR_USE_PREPROCESSING = True
    variations = engine._preprocess(frame)
    
    print(f"Generated {len(variations)} preprocessed variations.")
    
    for i, v in enumerate(variations):
        out_path = f"prep_var_{i}.jpg"
        cv2.imwrite(out_path, v)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image file")
    args = parser.parse_args()
    test_preprocessing(Path(args.image_path))
