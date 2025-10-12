import pytesseract
from PIL import Image
import os
import sys
from dotenv import load_dotenv

# Try to import PDF conversion library
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("[OCRAgent] Warning: pdf2image not installed. PDF support disabled.")
    print("[OCRAgent] Install with: pip install pdf2image")

# Load environment variables from .env file
load_dotenv()


class OCRAgent:
    def __init__(self, tesseract_cmd=None, device="cpu", mode="printed"):
        """
        Simple and reliable OCR Agent using Tesseract for printed text.
        
        Args:
            tesseract_cmd: Path to Tesseract executable (optional, auto-detects)
            device: Kept for compatibility with pipeline (not used by Tesseract)
            mode: Kept for compatibility with pipeline (not used by Tesseract)
        """
        # Priority order: 
        # 1. Explicit parameter
        # 2. Environment variable from .env
        # 3. Auto-detect (assumes in PATH)
        
        if tesseract_cmd is None:
            # Try to get from environment variable
            tesseract_cmd = os.getenv('TESSERACT_CMD')
            
            # If still None and on Windows, try common paths
            if tesseract_cmd is None and sys.platform == "win32":
                possible_paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                    r'C:\Users\pra21\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        tesseract_cmd = path
                        print(f"[OCRAgent] Auto-detected Tesseract at: {path}")
                        break
            
            # If still None, assume 'tesseract' is in PATH
            if tesseract_cmd is None:
                tesseract_cmd = "tesseract"
                print("[OCRAgent] Using 'tesseract' from PATH")
        
        # Set the Tesseract command path
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        print(f"[OCRAgent] Tesseract command: {tesseract_cmd}")
        
        # Verify Tesseract is working
        try:
            version = pytesseract.get_tesseract_version()
            print(f"[OCRAgent] ✓ Tesseract {version} ready")
        except Exception as e:
            print(f"[OCRAgent] ⚠ Warning: Could not verify Tesseract: {e}")
            print("[OCRAgent] Make sure Tesseract is in your PATH or provide correct path")
    
    def extract_text(self, image_path, lang='eng'):
        """
        Extracts text from an image using Tesseract OCR.
        
        Args:
            image_path (str): Path to the image file
            lang (str): Language code (eng, spa, fra, deu, etc.)
        
        Returns:
            str: Extracted text from the image
        """
        try:
            # Check if it's a PDF
            if image_path.lower().endswith('.pdf'):
                return self._extract_from_pdf(image_path, lang)
            
            # Regular image file
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang=lang)
            return text.strip()
        except Exception as e:
            return f"Error during OCR: {str(e)}"
    
    def _extract_from_pdf(self, pdf_path, lang='eng'):
        """
        Extract text from PDF by converting to images first.
        
        Args:
            pdf_path: Path to PDF file
            lang: Language code
        
        Returns:
            str: Extracted text from all pages
        """
        if not PDF_SUPPORT:
            return "Error: PDF support not available. Install pdf2image: pip install pdf2image"
        
        try:
            print(f"[OCRAgent] Converting PDF to images...")
            
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)
            
            print(f"[OCRAgent] Processing {len(images)} pages...")
            
            # Extract text from each page
            all_text = []
            for i, img in enumerate(images):
                print(f"[OCRAgent] Processing page {i+1}/{len(images)}...")
                page_text = pytesseract.image_to_string(img, lang=lang)
                if page_text.strip():
                    all_text.append(f"=== Page {i+1} ===\n{page_text.strip()}")
            
            return "\n\n".join(all_text)
            
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    
    def detect_text(self, input_path, lang='eng'):
        """
        Pipeline-compatible method that returns dict format.
        Supports both images and PDFs.
        
        Args:
            input_path: Path to image or PDF file
            lang: Language code for Tesseract (eng, spa, fra, deu, etc.)
        
        Returns:
            dict: {
                "extracted_text": str,
                "text_blocks": list,
                "annotated_image": PIL.Image,
                "method": str
            }
        """
        try:
            # Check file type
            is_pdf = input_path.lower().endswith('.pdf')
            
            if is_pdf:
                print(f"[OCRAgent] Detected PDF file")
            
            # Extract text
            text = self.extract_text(input_path, lang=lang)
            
            # For PDFs, get first page as image for preview
            if is_pdf and PDF_SUPPORT and not text.startswith("Error"):
                try:
                    images = convert_from_path(input_path, dpi=150, first_page=1, last_page=1)
                    img = images[0] if images else None
                except:
                    img = None
            else:
                # Regular image file
                try:
                    img = Image.open(input_path)
                except:
                    img = None
            
            # Log extraction info
            if not text.startswith("Error"):
                print(f"[OCRAgent] ✓ Extraction complete using tesseract!")
                print(f"[OCRAgent] Extracted text length: {len(text)} characters")
                
                if is_pdf:
                    pages = text.count("=== Page")
                    print(f"[OCRAgent] Processed {pages} page(s)")
                
                if len(text.strip()) < 10:
                    print(f"[OCRAgent] ⚠ Warning: Very short text extracted")
                    print(f"[OCRAgent] Extracted: '{text}'")
                    print(f"[OCRAgent] Consider:")
                    print(f"  - Check image/PDF quality and resolution")
                    print(f"  - Verify file contains readable text")
                    print(f"  - Try preprocessing the image")
                else:
                    # Show preview of extracted text
                    preview = text[:100] + "..." if len(text) > 100 else text
                    print(f"[OCRAgent] Preview: '{preview}'")
            
            # Return in pipeline-expected format
            return {
                "extracted_text": text,
                "text_blocks": [],  # Simple mode doesn't compute blocks
                "annotated_image": img,
                "method": "tesseract" + (" (PDF)" if is_pdf else "")
            }
        except Exception as e:
            print(f"[OCRAgent] ✗ Error during OCR: {e}")
            import traceback
            traceback.print_exc()
            return {
                "extracted_text": f"Error during OCR: {str(e)}",
                "text_blocks": [],
                "annotated_image": None,
                "method": "tesseract"
            }
    
    def detect_text_from_np(self, image_np, lang='eng'):
        """
        Alternative method for numpy array input (for pipeline compatibility).
        
        Args:
            image_np: Numpy array (BGR format from cv2.imread)
            lang: Language code
        
        Returns:
            dict with extracted_text, text_blocks, annotated_image, method
        """
        import tempfile
        import cv2
        
        # Save numpy array to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image_np)
            temp_path = tmp.name
        
        try:
            result = self.detect_text(temp_path, lang=lang)
            return result
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass


# Example usage
if __name__ == "__main__":
    import sys
    
    # Test 1: Auto-detect Tesseract
    print("="*60)
    print("Test 1: Auto-detect Tesseract")
    print("="*60)
    ocr = OCRAgent()
    
    # Test 2: Extract text from image
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = "test.jpg"
    
    if os.path.exists(test_image):
        print(f"\nTesting with image: {test_image}")
        print("-"*60)
        
        # Simple extraction
        text = ocr.extract_text(test_image)
        print("\nExtracted Text (simple method):")
        print("-"*60)
        print(text)
        
        # Pipeline-compatible method
        print("\n" + "="*60)
        print("Pipeline Format:")
        print("="*60)
        result = ocr.detect_text(test_image)
        print(f"Method: {result['method']}")
        print(f"Text length: {len(result['extracted_text'])} characters")
        print(f"\nText preview:")
        print(result['extracted_text'][:200] + "..." if len(result['extracted_text']) > 200 else result['extracted_text'])
    else:
        print(f"\n⚠ Test image not found: {test_image}")
        print("Usage: python ocr_agent.py <image_path>")