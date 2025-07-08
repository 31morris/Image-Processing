import cv2
import numpy as np
import os
import pytesseract

class SpotshadeProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = self._load_image()
        self.results_dir = "results"
        self._setup_results_dir()

    def _load_image(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {self.image_path}")
        return image

    def _setup_results_dir(self):
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def _get_output_filename(self, suffix):
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        return os.path.join(self.results_dir, f"{base_name}{suffix}")
    
    def write_image(self, suffix, img):
        output_path = self._get_output_filename(suffix)
        cv2.imwrite(output_path, img)
        print(f"Image saved to {output_path}")
        self.image = img

    def remove_shadow(self):
        bilateral_filter = cv2.bilateralFilter(self.image, d=9, sigmaColor=75, sigmaSpace=75)
        blurred_image = cv2.GaussianBlur(bilateral_filter, (5, 5), 0)
        kernel = np.ones((3, 3), np.uint8)
        morph_image = cv2.morphologyEx(blurred_image, cv2.MORPH_CLOSE, kernel)
        normalized_img = cv2.adaptiveThreshold(morph_image, 255, 
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
        self.write_image("_shadow_removed.png", normalized_img)

    def ocr_image(self):
        text = pytesseract.image_to_string(self.image, lang='eng')
        output_path = self._get_output_filename(".txt")
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Extracted text saved to {output_path}")

if __name__ == "__main__":
    image_file = "text-spotshade.tif"
    processor = SpotshadeProcessor(image_file)

    processor.remove_shadow()
    processor.ocr_image()
