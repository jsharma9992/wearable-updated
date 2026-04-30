"""
OCR Engine
──────────
Image preprocessing pipeline + EasyOCR execution.
Preprocessing: greyscale → upscale → denoise → CLAHE → adaptive
threshold → deskew.
"""

import cv2
import numpy as np
import logging
from config import OCR_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


class OCREngine:

    def __init__(self, easyocr_reader=None, paddle_reader=None):
        self._easyocr_reader = easyocr_reader
        self._paddle_reader = paddle_reader

    # ── public ──────────────────────────────────────────────

    def read_region(self, frame, bbox):
        """OCR a single (x,y,w,h) region. Returns (text, confidence)."""
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w <= 0 or h <= 0:
            return "", 0.0
        crop = frame[y:y + h, x:x + w]
        return self._ocr(self._preprocess(crop))

    def read_boxes(self, frame, text_boxes):
        """
        Read from a list of boxes.  Reuses text already extracted by
        the detector when available.

        Groups words into lines by Y proximity, then sorts each line
        left-to-right for natural reading order.

        Returns (combined_text, average_confidence).
        """
        # Extract text and position from each box
        words = []
        for box in text_boxes:
            text = None
            conf = 0.0

            if box.get("text") and box["confidence"] >= OCR_CONFIDENCE_THRESHOLD:
                text = box["text"]
                conf = box["confidence"]
            else:
                t, c = self.read_region(frame, box["bbox"])
                if t and c >= OCR_CONFIDENCE_THRESHOLD:
                    text = t
                    conf = c

            if text:
                x, y, w, h = box["bbox"]
                words.append({
                    "text": text,
                    "conf": conf,
                    "x": x,
                    "y": y,
                    "h": h,
                })

        if not words:
            return "", 0.0

        # Calculate line grouping threshold based on average word height
        avg_h = max(np.mean([w["h"] for w in words]), 10)
        line_threshold = avg_h * 0.6  # words within 60% of avg height = same line

        # Sort by Y first
        words.sort(key=lambda w: w["y"])

        # Group into lines by Y proximity
        lines = [[words[0]]]
        for word in words[1:]:
            # Compare with the average Y of the current line
            line_avg_y = np.mean([w["y"] for w in lines[-1]])
            if abs(word["y"] - line_avg_y) <= line_threshold:
                lines[-1].append(word)
            else:
                lines.append([word])

        # Sort each line by X (left to right), then join
        all_texts = []
        all_confs = []
        for line in lines:
            line.sort(key=lambda w: w["x"])
            for w in line:
                all_texts.append(w["text"])
                all_confs.append(w["conf"])

        combined = " ".join(all_texts)
        avg = float(np.mean(all_confs)) if all_confs else 0.0
        return combined, avg

    # ── preprocessing ───────────────────────────────────────

    @staticmethod
    def _is_clean_image(gray):
        """
        Detect if image is already clean/high-contrast (e.g. printed book page).
        Returns True if image looks clean and doesn't need heavy processing.
        """
        std = np.std(gray)
        mean_val = np.mean(gray)

        # Photographed book pages in warm/indoor lighting have mean ~120-180
        # and moderate-to-high std. Lower thresholds to catch these.
        if (mean_val > 120 and std > 30) or std > 50:
            return True
        return False

    def _preprocess(self, img):
        """Returns a list of preprocessed images to try."""
        import config
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

        # upscale tiny crops
        h, w = gray.shape
        if h < 50 or w < 50:
            s = max(50 / h, 50 / w, 2.0)
            gray = cv2.resize(gray, None, fx=s, fy=s,
                              interpolation=cv2.INTER_CUBIC)

        variations = [gray] # Original (grayscale)
        
        if not config.OCR_USE_PREPROCESSING:
            return variations

        # Enhanced (Denoise + CLAHE + Sharpen)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        variations.append(sharpened)

        # Thresholded (Adaptive/Otsu)
        binary = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2
        )
        binary = cv2.medianBlur(binary, 3)
        variations.append(binary)
        
        return [self._deskew(v) for v in variations]

    @staticmethod
    def _deskew(img):
        coords = np.column_stack(np.where(img < 128))
        if len(coords) < 50:
            return img
        try:
            angle = cv2.minAreaRect(coords)[-1]
            angle = -(90 + angle) if angle < -45 else -angle
            if abs(angle) > 15 or abs(angle) < 0.5:
                return img
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            return cv2.warpAffine(img, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            return img

    # ── recognition ─────────────────────────────────────────

    def _ocr(self, images):
        """OCR on preprocessed images. Tries multiple, picks best."""
        import config
        best_text = ""
        best_conf = 0.0

        for img in images:
            combined, avg = self._ocr_single_image(img)
            
            # Simple scoring: longer text + better confidence wins
            # You can tweak this heuristic
            score_current = len(combined) * avg
            score_best = len(best_text) * best_conf
            
            if score_current > score_best:
                best_text = combined
                best_conf = avg
                
        return best_text, best_conf

    def _ocr_single_image(self, image):
        """Run PaddleOCR, fallback to EasyOCR"""
        import config
        
        # 1. Try PaddleOCR
        if config.OCR_PRIMARY == "paddle" and self._paddle_reader is not None:
            try:
                results = self._paddle_reader.ocr(image, cls=False)
                texts = []
                confs = []
                if results and results[0]:
                    for line in results[0]:
                        if not line or len(line) < 2: continue
                        text, conf = line[1]
                        if conf >= config.OCR_MIN_CONFIDENCE:
                            texts.append(text)
                            confs.append(conf)
                combined = " ".join(texts)
                avg = float(np.mean(confs)) if confs else 0.0
                if combined:
                    return combined, avg
            except Exception as exc:
                logger.error("PaddleOCR failed: %s, falling back to EasyOCR", exc)

        # 2. Try EasyOCR fallback
        if self._easyocr_reader is None:
            return "", 0.0

        try:
            results = self._easyocr_reader.readtext(image, detail=1, paragraph=True)

            texts = []
            confs = []

            for item in results:

                if len(item) == 3:
                    _, text, conf = item
                else:
                    text = item[1]
                    conf = 0.5

                if conf >= config.OCR_CONFIDENCE_THRESHOLD:
                    texts.append(text)
                    confs.append(conf)

            combined = " ".join(texts)
            avg = float(np.mean(confs)) if confs else 0.0

            return combined, avg

        except Exception as exc:
            logger.error("OCR failed: %s", exc)
            return "", 0.0

    def _run_readtext(self, image):
        """
        Run OCR (Paddle or EasyOCR fallback) and reconstruct reading order.
        Returns (text, confidence).
        """
        import config
        
        parsed = []
        
        # 1. PaddleOCR
        if config.OCR_PRIMARY == "paddle" and self._paddle_reader is not None:
            try:
                results = self._paddle_reader.ocr(image, cls=False)
                if results and results[0]:
                    for line in results[0]:
                        if not line or len(line) < 2: continue
                        bbox, (text, conf) = line
                        if conf >= config.OCR_MIN_CONFIDENCE and text.strip():
                            pts = np.array(bbox)
                            y_center = np.mean(pts[:, 1])
                            x_left = np.min(pts[:, 0])
                            parsed.append({
                                "text": text.strip(),
                                "conf": conf,
                                "y": y_center,
                                "x": x_left,
                            })
            except Exception as exc:
                logger.error("PaddleOCR failed in _run_readtext: %s", exc)
                parsed = [] # reset for fallback
        
        # 2. EasyOCR Fallback
        if not parsed and self._easyocr_reader is not None:
            try:
                results = self._easyocr_reader.readtext(image, detail=1, paragraph=False)

                if results:
                    for item in results:
                        if len(item) == 3:
                            bbox, text, conf = item
                        elif len(item) == 2:
                            bbox, text = item
                            conf = 0.5
                        else:
                            continue

                        if conf >= config.OCR_CONFIDENCE_THRESHOLD and text.strip():
                            pts = np.array(bbox)
                            y_center = np.mean(pts[:, 1])
                            x_left = np.min(pts[:, 0])
                            parsed.append({
                                "text": text.strip(),
                                "conf": conf,
                                "y": y_center,
                                "x": x_left,
                            })
            except Exception as exc:
                logger.error("EasyOCR failed in _run_readtext: %s", exc)

        if not parsed:
            return "", 0.0

        # Group into lines: words with similar Y are on the same line
        parsed.sort(key=lambda w: w["y"])
        lines = []
        current_line = [parsed[0]]

        for word in parsed[1:]:
            # If Y difference is small (< 15px), same line
            if abs(word["y"] - current_line[-1]["y"]) < 15:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
        lines.append(current_line)

        # Sort words within each line by X (left to right)
        text_lines = []
        all_confs = []
        for line in lines:
            line.sort(key=lambda w: w["x"])
            line_text = " ".join(w["text"] for w in line)
            text_lines.append(line_text)
            all_confs.extend(w["conf"] for w in line)

        full_text = " ".join(text_lines)
        avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

        return full_text, avg_conf

    def read_full(self, frame):
        """
        Full-frame OCR for natural reading order.
        """
        # Try original frame
        raw_text, raw_conf = self._run_readtext(frame)

        if len(raw_text.strip()) >= 10:
            return raw_text, raw_conf

        # Try enhanced frame
        import config
        if config.OCR_USE_PREPROCESSING:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            light = clahe.apply(gray)
            light_text, light_conf = self._run_readtext(light)

            if len(light_text.strip()) > len(raw_text.strip()):
                return light_text, light_conf

        return raw_text, raw_conf