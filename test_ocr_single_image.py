import cv2
import argparse
from pathlib import Path
from intelligence.ocr_engine import OCREngine
from utils.logger import setup_logger
import config

setup_logger()

def test_ocr(image_path):
    print(f"Testing OCR on: {image_path}")
    
    paddle_reader = None
    easyocr_reader = None
    
    if config.OCR_PRIMARY == "paddle":
        print("Loading PaddleOCR...")
        try:
            from paddleocr import PaddleOCR
            import logging as plogging
            plogging.getLogger("ppocr").setLevel(plogging.ERROR)
            paddle_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        except ImportError:
            print("PaddleOCR not installed")
            
    print("Loading EasyOCR...")
    try:
        import easyocr
        easyocr_reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=config.OCR_GPU, verbose=False)
    except ImportError:
        print("EasyOCR not installed")
    
    engine = OCREngine(easyocr_reader=easyocr_reader, paddle_reader=paddle_reader)
    
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Could not load image: {image_path}")
        return
        
    text, conf = engine.read_full(frame)
    print("\n=== OCR Result ===")
    print(f"Confidence: {conf:.2f}")
    print(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image file")
    args = parser.parse_args()
    test_ocr(Path(args.image_path))
