import fitz  # PyMuPDF
import re
import cv2
import numpy as np
import streamlit as st
from io import BytesIO

# Function to find specific details using regular expressions
def find_detail(text, pattern, group=1):
    match = re.search(pattern, text)
    return match.group(group) if match else "Not found"

# Function to find white text and its bounding box
def find_white_text(page):
    text_instances = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    # Checking if the text color is white (1.0 for R, G, B in PDF context)
                    if span["color"] == 1.0:
                        text_instances.append({
                            "text": span["text"],
                            "bbox": span["bbox"]
                        })
    return text_instances

st.title("SP Energy Networks Processing App")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Read the PDF file
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    filename = uploaded_file.name

    # Patterns to search for specific details
    patterns = {
        "Date Requested": r'Date Requested:\s*(\S+)',
        "Job Reference": r'Job Reference:\s*(\S+)',
        "Site Location": r'Site Location:\s*([\d\s]+)',
        "Your Scheme/Reference": r'Your Scheme/Reference:\s*(\S+)',
        "Gas Warning": r'WARNING! This area contains (.*)'
    }

    # Loop through only the first 4 pages in the PDF
    for page_number in range(min(5, len(pdf_document))):
        st.write(f"\nProcessing Page {page_number + 1}")

        # Load the page
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()

        # Handle different numbers of color channels
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # Convert grayscale or RGBA to RGB
        if pix.n == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 1:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Extract text from the page
        text = page.get_text()

        # Extract and display the details
        details = {key: find_detail(text, pattern) for key, pattern in patterns.items()}
        for key, value in details.items():
            st.write(f"{key}: {value}")

        # Find white text
        white_texts = find_white_text(page)

        # Draw bounding boxes around white text
        for white_text in white_texts:
            x_min, y_min, x_max, y_max = map(int, white_text["bbox"])
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red bounding box

        # Convert hex to RGB
        hex_color = '#bebe49'
        rgb_color = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]

        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the range of the purple color in HSV
        lower_purple = np.array([140, 50, 50])
        upper_purple = np.array([160, 255, 255])

        # Create a mask for the purple color
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

        # Convert RGB to HSV
        rgb_color = np.uint8([[rgb_color]])
        hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)[0][0]

        tolerance = 4
        lower_color = np.array([hsv_color[0] - tolerance, 50, 50])
        upper_color = np.array([hsv_color[0] + tolerance, 255, 255])

        mask_IDNO = cv2.inRange(hsv, lower_color, upper_color)

        # Combine purple with each specified color
        combined_masks = {
            "SP Energy Networks.": cv2.bitwise_or(mask_IDNO, mask_purple),
        }

        # Manually define the bounding box coordinates (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = 8, 10, 585, 580

        # Apply bounding box mask to each combined mask
        combined_mask_page = np.zeros_like(mask_purple)
        for combined_mask in combined_masks.values():
            bbox_mask = np.zeros_like(combined_mask)
            bbox_mask[y_min:y_max, x_min:x_max] = combined_mask[y_min:y_max, x_min:x_max]
            combined_mask_page = cv2.bitwise_or(combined_mask_page, bbox_mask)

        # Apply the bounding box mask to the original image for this page
        result_page = cv2.bitwise_and(img, img, mask=combined_mask_page)

        # Draw the bounding box on the result image for this page
        result_with_bbox_page = result_page.copy()
        cv2.rectangle(result_with_bbox_page, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Display the result for this page
        st.image(cv2.cvtColor(result_with_bbox_page, cv2.COLOR_BGR2RGB), caption=f'Page {page_number + 1}', use_column_width=True)
