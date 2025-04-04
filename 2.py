import numpy as np
import cv2
import pandas as pd
import os
from PIL import Image
try:
    import easyocr
except ImportError:
    print("EasyOCR not installed. Installing now...")
    import subprocess
    subprocess.check_call(["pip", "install", "easyocr"])
    import easyocr

def preprocess_image(image_path):
    """
    Preprocess image to enhance text visibility for OCR
    """
    # Read image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    return img, denoised

def detect_table_structure(img):
    """
    Detect lines to identify table structure
    """
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    # Apply threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combine horizontal and vertical lines
    table_structure = cv2.add(horizontal_lines, vertical_lines)
    
    # Find contours
    contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour (likely the table)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h
    
    return 0, 0, img.shape[1], img.shape[0]

def find_cells(processed_img):
    """
    Find individual cells in the table
    """
    # Find contours
    contours, _ = cv2.findContours(~processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area to find table cells
    cells = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Filter by size to exclude very small or large contours
        if 100 < area < processed_img.shape[0] * processed_img.shape[1] / 4:
            cells.append((x, y, w, h))
    
    # Sort cells by row and column
    cells.sort(key=lambda cell: (cell[1], cell[0]))
    
    return cells

def extract_text_with_easyocr(img, x, y, w, h):
    """
    Extract text from a cell using EasyOCR
    """
    reader = easyocr.Reader(['en'])
    
    # Crop the cell
    cell_img = img[y:y+h, x:x+w]
    
    # Extract text
    results = reader.readtext(cell_img)
    
    # Combine all text found
    text = ' '.join([result[1] for result in results])
    
    return text

def extract_table_from_image(image_path, output_csv=None):
    """
    Main function to extract tabular data from an image
    """
    print(f"Processing image: {image_path}")
    
    # Preprocess image
    original_img, processed_img = preprocess_image(image_path)
    
    # Detect table area
    x, y, w, h = detect_table_structure(processed_img)
    table_img = processed_img[y:y+h, x:x+w]
    original_table_img = original_img[y:y+h, x:x+w]
    
    # Find cells
    cells = find_cells(table_img)
    
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'])
    print("EasyOCR initialized")
    
    # Group cells into rows based on y-coordinate
    rows = []
    current_row = []
    current_y = None
    
    # Sort cells by y-coordinate
    cells.sort(key=lambda cell: cell[1])
    
    # Group cells into rows
    for cell in cells:
        x, y, w, h = cell
        if current_y is None:
            current_y = y
            current_row.append(cell)
        elif abs(y - current_y) < 20:  # Cells in same row if y difference is small
            current_row.append(cell)
        else:
            # Sort cells in row by x-coordinate
            current_row.sort(key=lambda cell: cell[0])
            rows.append(current_row)
            current_row = [cell]
            current_y = y
    
    # Add the last row
    if current_row:
        current_row.sort(key=lambda cell: cell[0])
        rows.append(current_row)
    
    # Extract text from each cell and organize into a table
    table_data = []
    for row in rows:
        row_data = []
        for cell in row:
            x, y, w, h = cell
            # Use original image for better OCR results
            cell_img = original_table_img[y:y+h, x:x+w]
            
            # OCR using EasyOCR
            results = reader.readtext(cell_img)
            cell_text = ' '.join([result[1] for result in results])
            row_data.append(cell_text.strip())
        
        # Only add non-empty rows
        if any(cell.strip() for cell in row_data):
            table_data.append(row_data)
    
    # Create DataFrame
    # First row may contain headers, but in case of partial detection, we'll use generic column names
    if table_data:
        # Check if all rows have the same number of columns
        max_cols = max(len(row) for row in table_data)
        for i in range(len(table_data)):
            while len(table_data[i]) < max_cols:
                table_data[i].append("")
    
        # Use first row as headers if it seems to contain header information
        headers = []
        if len(table_data) > 1:
            potential_headers = table_data[0]
            if all(isinstance(header, str) and header.strip() for header in potential_headers):
                headers = potential_headers
                table_data = table_data[1:]
    
        # If no valid headers are found, use generic column names
        if not headers:
            headers = [f"Column {i+1}" for i in range(max_cols)]
    
        df = pd.DataFrame(table_data, columns=headers)
        
        # Save to CSV if output path is provided
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Table data saved to {output_csv}")
        
        return df
    else:
        print("No table data detected.")
        return None

# Example usage
if __name__ == "__main__":
    image_path = "table_image.jpg"  # Replace with your image path
    output_csv = "extracted_table.csv"  # Output CSV path
    
    df = extract_table_from_image(image_path, output_csv)
    if df is not None:
        print("\nExtracted Data Preview:")
        print(df.head())
