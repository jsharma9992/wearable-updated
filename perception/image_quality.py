import cv2
import numpy as np

class ImageQualityChecker:
    @staticmethod
    def evaluate(frame):
        """
        Evaluate image quality.
        Returns:
            dict with 'blur', 'brightness', 'contrast'
            and 'is_good' boolean.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        
        # Blur score using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness score (average pixel intensity)
        brightness_score = np.mean(gray)
        
        # Contrast score (RMS contrast)
        contrast_score = np.std(gray)
        
        return {
            "blur": blur_score,
            "brightness": brightness_score,
            "contrast": contrast_score
        }
