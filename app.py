import streamlit as st
import cv2
import easyocr
import numpy as np
from PIL import Image
import io
import time

# Set page configuration
st.set_page_config(
    page_title="Number Plate Detector",
    page_icon="🚗",
    layout="wide"
)

class NumberPlateDetector:
    def __init__(self):
        self.reader = None
    
    @st.cache_resource
    def initialize_reader(_self):
        """Initialize OCR reader (cached to avoid reloading)"""
        return easyocr.Reader(['en'])
    
    def get_reader(self):
        if self.reader is None:
            self.reader = self.initialize_reader()
        return self.reader
    
    def detect_plates_simple(self, image):
        """Simple OCR-based detection"""
        reader = self.get_reader()
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Ensure RGB format
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_rgb = img_array
        else:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        result_img = img_rgb.copy()
        results = reader.readtext(img_rgb)
        detected_texts = []
        
        for (bbox, text, confidence) in results:
            # Filter for potential license plates
            if confidence > 0.5 and len(text.strip()) >= 3:
                # Check if text contains alphanumeric characters
                if any(c.isalnum() for c in text):
                    detected_texts.append({
                        'text': text.strip(),
                        'confidence': confidence,
                        'bbox': bbox,
                        'method': 'Simple OCR'
                    })
                    
                    # Draw bounding box
                    points = np.array(bbox, dtype=np.int32)
                    cv2.polylines(result_img, [points], True, (0, 255, 0), 3)
                    
                    # Add text label
                    cv2.putText(result_img, f"{text.strip()}", 
                              (int(bbox[0][0]), int(bbox[0][1] - 10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return result_img, detected_texts
    
    def detect_plates_advanced(self, image):
        """Advanced contour-based detection"""
        reader = self.get_reader()
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Convert to proper format
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_rgb = img_array
        else:
            img_bgr = img_array
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply image processing
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        
        result_img = img_rgb.copy()
        detected_texts = []
        
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h
                
                # Filter by size and aspect ratio
                if 1.5 <= aspect_ratio <= 6.0 and w > 60 and h > 20:
                    # Extract plate region
                    plate_region = img_rgb[y:y+h, x:x+w]
                    
                    if plate_region.size > 0:
                        results = reader.readtext(plate_region)
                        
                        for (bbox, text, confidence) in results:
                            if confidence > 0.4 and len(text.strip()) >= 3:
                                detected_texts.append({
                                    'text': text.strip(),
                                    'confidence': confidence,
                                    'region': (x, y, w, h),
                                    'method': 'Advanced Contour'
                                })
                                
                                # Draw rectangle
                                cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
                                cv2.putText(result_img, f"{text.strip()}", 
                                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        return result_img, detected_texts

def main():
    # Title and header
    st.title("🚗 Number Plate Detector")
    st.markdown("### Upload an image to detect and extract license plate numbers!")
    
    # Sidebar
    st.sidebar.header("Settings")
    detection_method = st.sidebar.selectbox(
        "Choose Detection Method:",
        ["Simple OCR", "Advanced Contour", "Both Methods"]
    )
    
    confidence_threshold = st.sidebar.slider(
        "Minimum Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Lower values detect more text but may include false positives"
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload an image containing a vehicle with visible license plate"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Initialize detector
        detector = NumberPlateDetector()
        
        # Process button
        if st.button("🔍 Detect License Plates", type="primary"):
            with st.spinner("Processing image... This may take a moment..."):
                
                # Initialize OCR reader if not already done
                if detector.reader is None:
                    progress_bar = st.progress(0)
                    st.info("Initializing OCR reader for the first time...")
                    detector.get_reader()
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    progress_bar.empty()
                
                results = []
                
                # Run selected detection methods
                if detection_method in ["Simple OCR", "Both Methods"]:
                    result_img1, detected_texts1 = detector.detect_plates_simple(image)
                    results.extend(detected_texts1)
                
                if detection_method in ["Advanced Contour", "Both Methods"]:
                    result_img2, detected_texts2 = detector.detect_plates_advanced(image)
                    results.extend(detected_texts2)
                
                # Display results
                with col2:
                    st.subheader("Detection Results")
                    
                    if detection_method == "Simple OCR":
                        st.image(result_img1, caption="Simple OCR Detection (Green boxes)", use_column_width=True)
                    elif detection_method == "Advanced Contour":
                        st.image(result_img2, caption="Advanced Contour Detection (Blue boxes)", use_column_width=True)
                    else:  # Both methods
                        # Combine both images
                        combined_img = result_img1.copy()
                        # Add blue boxes from advanced method
                        for detection in detected_texts2:
                            if 'region' in detection:
                                x, y, w, h = detection['region']
                                cv2.rectangle(combined_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
                        st.image(combined_img, caption="Combined Detection (Green: Simple, Blue: Advanced)", use_column_width=True)
                
                # Filter results by confidence
                filtered_results = [r for r in results if r['confidence'] >= confidence_threshold]
                
                # Remove duplicates (same text with similar confidence)
                unique_results = []
                for result in filtered_results:
                    is_duplicate = False
                    for existing in unique_results:
                        if (result['text'].lower() == existing['text'].lower() and 
                            abs(result['confidence'] - existing['confidence']) < 0.1):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_results.append(result)
                
                # Display detected text
                st.subheader("📋 Detected License Plates")
                
                if unique_results:
                    st.success(f"Found {len(unique_results)} license plate(s)!")
                    
                    # Create a table of results
                    for i, detection in enumerate(unique_results, 1):
                        with st.container():
                            col_a, col_b, col_c = st.columns([2, 1, 1])
                            
                            with col_a:
                                st.markdown(f"**{i}. {detection['text']}**")
                            
                            with col_b:
                                confidence_color = "🟢" if detection['confidence'] > 0.8 else "🟡" if detection['confidence'] > 0.6 else "🟠"
                                st.markdown(f"{confidence_color} {detection['confidence']:.1%}")
                            
                            with col_c:
                                method_emoji = "🔤" if detection['method'] == "Simple OCR" else "🔍"
                                st.markdown(f"{method_emoji} {detection['method']}")
                    
                    # Download results
                    results_text = "\n".join([
                        f"License Plate {i}: {detection['text']} (Confidence: {detection['confidence']:.1%}, Method: {detection['method']})"
                        for i, detection in enumerate(unique_results, 1)
                    ])
                    
                    st.download_button(
                        label="📥 Download Results",
                        data=results_text,
                        file_name="license_plate_results.txt",
                        mime="text/plain"
                    )
                    
                else:
                    st.warning("No license plates detected. Try:")
                    st.markdown("""
                    - Lowering the confidence threshold in the sidebar
                    - Using a clearer image with better lighting
                    - Ensuring the license plate is clearly visible and not too small
                    - Trying a different detection method
                    """)
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload an image to get started!")
        
        # Example images section
        st.markdown("---")
        st.subheader("💡 Tips for Best Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🎯 Good Image Quality**")
            st.markdown("""
            - Clear, well-lit images
            - License plate clearly visible
            - Minimal blur or distortion
            """)
        
        with col2:
            st.markdown("**📐 Proper Perspective**")
            st.markdown("""
            - Front or rear view of vehicle
            - License plate not at extreme angles
            - Plate not partially obscured
            """)
        
        with col3:
            st.markdown("**⚙️ Settings**")
            st.markdown("""
            - Try different detection methods
            - Adjust confidence threshold
            - Use 'Both Methods' for best coverage
            """)

if __name__ == "__main__":
    main()