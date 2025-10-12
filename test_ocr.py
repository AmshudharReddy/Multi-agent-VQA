"""
OCR Testing Script - Test different modes and see results
"""
import cv2
import sys
sys.path.append('.')

from agents.ocr_agent import OCRAgent

def test_ocr(image_path):
    """Test OCR with all three modes"""
    
    print("="*60)
    print(f"Testing OCR on: {image_path}")
    print("="*60)
    
    # Load image for inspection
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"✓ Image loaded: {img.shape[1]}x{img.shape[0]} pixels")
    print()
    
    # Test 1: Auto Mode
    print("🤖 TEST 1: AUTO MODE (Smart Detection)")
    print("-"*60)
    try:
        ocr_auto = OCRAgent(device="cpu", mode="auto")
        result_auto = ocr_auto.detect_text(image_path)
        print(f"Method used: {result_auto['method']}")
        print(f"Text found: '{result_auto['extracted_text']}'")
        print(f"Number of regions: {len(result_auto['text_blocks'])}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Test 2: Printed Mode (Tesseract)
    print("📄 TEST 2: PRINTED MODE (Tesseract Only)")
    print("-"*60)
    try:
        ocr_printed = OCRAgent(device="cpu", mode="printed")
        result_printed = ocr_printed.detect_text(image_path)
        print(f"Method used: {result_printed['method']}")
        print(f"Text found: '{result_printed['extracted_text']}'")
        print(f"Number of regions: {len(result_printed['text_blocks'])}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Test 3: Handwritten Mode (TrOCR)
    print("✍️ TEST 3: HANDWRITTEN MODE (TrOCR Only)")
    print("-"*60)
    try:
        ocr_handwritten = OCRAgent(device="cpu", mode="handwritten")
        result_handwritten = ocr_handwritten.detect_text(image_path)
        print(f"Method used: {result_handwritten['method']}")
        print(f"Text found: '{result_handwritten['extracted_text']}'")
        print(f"Number of regions: {len(result_handwritten['text_blocks'])}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Comparison
    print("="*60)
    print("📊 COMPARISON")
    print("="*60)
    try:
        print(f"Auto mode:        '{result_auto['extracted_text']}'")
    except:
        print("Auto mode:        Failed")
    
    try:
        print(f"Printed mode:     '{result_printed['extracted_text']}'")
    except:
        print("Printed mode:     Failed")
    
    try:
        print(f"Handwritten mode: '{result_handwritten['extracted_text']}'")
    except:
        print("Handwritten mode: Failed")
    
    print("="*60)
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    try:
        auto_len = len(result_auto['extracted_text'].strip())
        printed_len = len(result_printed['extracted_text'].strip())
        
        if printed_len > auto_len:
            print("   → Use 'printed' mode for better results")
        elif auto_len > 10:
            print("   → Auto mode is working well")
        else:
            print("   → Check image quality:")
            print("     • Is text clearly visible?")
            print("     • Is resolution high enough?")
            print("     • Is image properly oriented?")
    except:
        pass
    
    print()

def quick_test(image_path, mode="auto"):
    """Quick single-mode test"""
    print(f"Testing {mode} mode on {image_path}...")
    ocr = OCRAgent(device="cpu", mode=mode)
    result = ocr.detect_text(image_path)
    print(f"\n✓ Text: '{result['extracted_text']}'")
    print(f"✓ Method: {result['method']}")
    print(f"✓ Regions: {len(result['text_blocks'])}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Full test:  python test_ocr.py image.jpg")
        print("  Quick test: python test_ocr.py image.jpg [auto|printed|handwritten]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if len(sys.argv) == 3:
        # Quick test with specific mode
        mode = sys.argv[2]
        quick_test(image_path, mode)
    else:
        # Full comparison test
        test_ocr(image_path)