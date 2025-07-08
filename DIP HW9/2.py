# repair_broken_image_w.py
import cv2
import os
import pytesseract

class ImageProcessor:
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

    def repair_broken_image_w(self):
        _, binary = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        eroded = cv2.erode(binary, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        self.write_image("_proc.png", dilated)

    def ocr_image(self):
        text = pytesseract.image_to_string(self.image, lang='eng')
        output_path = self._get_output_filename(".txt")
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Extracted text saved to {output_path}")

if __name__ == "__main__":
    processor = ImageProcessor("text-broken.tif")
    processor.repair_broken_image_w()
    processor.ocr_image()
