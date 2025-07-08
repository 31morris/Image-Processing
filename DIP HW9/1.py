# process_image_edge.py
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

    def process_image_edge(self):
        _, binary = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = binary.shape
        for contour in contours:
            x, _, w, _ = cv2.boundingRect(contour)
            if x + w >= width:
                cv2.drawContours(binary, [contour], -1, (0), thickness=cv2.FILLED)
        self.write_image("_proc.png", binary)

    def ocr_image(self):
        text = pytesseract.image_to_string(self.image, lang='eng')
        output_path = self._get_output_filename(".txt")
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Extracted text saved to {output_path}")

if __name__ == "__main__":
    processor = ImageProcessor("text.tif")
    processor.process_image_edge()
    processor.ocr_image()
