import numpy as np
import cv2
import pandas as pd
import math
from PIL import Image
import os

try:
    import easyocr
except ImportError:
    print("EasyOCR not installed. Installing now...")
    import subprocess
    subprocess.check_call(["pip", "install", "easyocr"])
    import easyocr

def detect_rotation_angle(image):
    """
    Detect the rotation angle of the image using line detection
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use HoughLines to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    
    if lines is None:
        return 0
    
    # Calculate the most common angle
    angles = []
    for line in lines:
        rho, theta = line[0]
        # Convert theta to degrees and normalize to [-90, 90]
        angle_deg = np.degrees(theta) - 90
        if angle_deg < -90:
            angle_deg += 180
        elif angle_deg > 90:
            angle_deg -= 180
        angles.append(angle_deg)
    
    # Get the most common angle (simple approach)
    angle_counts = {}
    for angle in angles:
        # Round to nearest degree
        rounded = round(angle)
        if rounded in angle_counts:
            angle_counts[rounded] += 1
        else:
            angle_counts[rounded] = 1
    
    if not angle_counts:
        return 0
        
    # Find the most common angle
    most_common_angle = max(angle_counts.items(), key=lambda x: x[1])[0]
    
    # If the angle is close to horizontal or vertical (0, 90, -90), return 0
    if abs(most_common_angle) < 1 or abs(abs(most_common_angle) - 90) < 1:
        return 0
    
    return most_common_angle

def rotate_image(image, angle):
    """
    Rotate the image by the given angle
    """
    # Get image dimensions
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Calculate rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)
    
    # Adjust rotation matrix
    rotation_matrix[0, 2] += new_width / 2 - center[0]
    rotation_matrix[1, 2] += new_height / 2 - center[1]
    
    # Perform rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, 
                            borderValue=(255, 255, 255))
    
    return rotated

def preprocess_image(image_path):
    """
    Preprocess image including rotation correction
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Detect rotation angle
    angle = detect_rotation_angle(img)
    print(f"Detected rotation angle: {angle} degrees")
    
    # Rotate image if needed
    if abs(angle) > 1:
        img = rotate_image(img, angle)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    return img, denoised

def find_table_grid(img):
    """
    Find table grid lines for better cell detection
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Apply threshold
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combine lines
    table_grid = cv2.add(horizontal_lines, vertical_lines)
    
    # Dilate to connect components
    kernel = np.ones((3, 3), np.uint8)
    table_grid = cv2.dilate(table_grid, kernel, iterations=1)
    
    return table_grid

def find_table_cells(grid_img):
    """
    Find cells in the table grid
    """
    # Find contours
    contours, _ = cv2.findContours(grid_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area - largest contour is likely the table boundary
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Skip the first contour (table boundary) if there are multiple contours
    cell_contours = contours[1:] if len(contours) > 1 else contours
    
    # Extract bounding boxes for cells
    cells = []
    min_area = 100  # Filter out very small contours
    
    for contour in cell_contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        if area > min_area:
            cells.append((x, y, w, h))
    
    # Sort cells by row then column (top to bottom, left to right)
    # First, identify rows based on y-coordinate proximity
    if not cells:
        return []
        
    # Sort by y-coordinate
    cells.sort(key=lambda c: c[1])
    
    # Group cells into rows
    rows = []
    current_row = [cells[0]]
    row_top = cells[0][1]
    
    for cell in cells[1:]:
        y_diff = abs(cell[1] - row_top)
        if y_diff < 20:  # Cells in the same row if y is within 20 pixels
            current_row.append(cell)
        else:
            # Sort current row by x-coordinate
            current_row.sort(key=lambda c: c[0])
            rows.append(current_row)
            # Start new row
            current_row = [cell]
            row_top = cell[1]
    
    # Add the last row
    if current_row:
        current_row.sort(key=lambda c: c[0])
        rows.append(current_row)
    
    return rows

def extract_table_from_image(image_path, output_csv=None):
    """
    Enhanced function to extract tabular data from a rotated image
    """
    print(f"Processing image: {image_path}")
    
    # Initialize EasyOCR once (to avoid reloading for each cell)
    reader = easyocr.Reader(['en'])
    print("EasyOCR initialized")
    
    # Preprocess image (includes rotation correction)
    original_img, processed_img = preprocess_image(image_path)
    
    # Find table grid
    grid = find_table_grid(processed_img)
    
    # Save the processed image and grid for debugging
    cv2.imwrite("processed_image.jpg", processed_img)
    cv2.imwrite("table_grid.jpg", grid)
    
    # Find rows of cells
    rows = find_table_cells(grid)
    
    if not rows:
        print("No table structure detected.")
        return None
    
    # Extract text from each cell
    table_data = []
    header_row = None
    
    for row_idx, row in enumerate(rows):
        row_data = []
        for cell in row:
            x, y, w, h = cell
            # Add some padding around the cell
            padding = 2
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = w + 2 * padding
            h_pad = h + 2 * padding
            
            # Ensure we don't go out of bounds
            if x_pad + w_pad > original_img.shape[1]:
                w_pad = original_img.shape[1] - x_pad
            if y_pad + h_pad > original_img.shape[0]:
                h_pad = original_img.shape[0] - y_pad
                
            cell_img = original_img[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            
            # Skip empty cells
            if cell_img.size == 0:
                row_data.append("")
                continue
                
            # Save cell image for debugging
            if row_idx < 2 and len(row_data) < 5:  # Save just a few cells
                cv2.imwrite(f"cell_{row_idx}_{len(row_data)}.jpg", cell_img)
            
            # Extract text using EasyOCR
            try:
                results = reader.readtext(cell_img)
                cell_text = ' '.join([result[1] for result in results])
                row_data.append(cell_text.strip())
            except Exception as e:
                print(f"Error in OCR: {e}")
                row_data.append("")
        
        # Only add non-empty rows
        if any(cell.strip() for cell in row_data):
            # If this is the first non-empty row, it's likely the header
            if header_row is None:
                header_row = row_data
            else:
                table_data.append(row_data)
    
    # Create DataFrame
    if table_data and header_row:
        # Try to identify if the first row is actually a valid header
        # A valid header typically doesn't contain numbers as the majority of cells
        numeric_cells = sum(1 for cell in header_row if any(c.isdigit() for c in cell))
        is_header_numeric = numeric_cells > len(header_row) / 2
        
        if is_header_numeric:
            # If the header row is mostly numbers, it's probably data
            table_data.insert(0, header_row)
            # Create generic column names
            max_cols = max(len(row) for row in table_data)
            column_names = ["Column " + str(i+1) for i in range(max_cols)]
        else:
            # Use detected header
            column_names = header_row
            
        # Ensure all rows have the same number of columns
        max_cols = len(column_names)
        for i in range(len(table_data)):
            while len(table_data[i]) < max_cols:
                table_data[i].append("")
            # Truncate if longer than header
            if len(table_data[i]) > max_cols:
                table_data[i] = table_data[i][:max_cols]
        
        # Create DataFrame
        df = pd.DataFrame(table_data, columns=column_names)
        
        # Save to CSV
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Table data saved to {output_csv}")
        
        return df
    else:
        print("No valid table data detected.")
        return None

# Example usage
if __name__ == "__main__":
    image_path = "table_image.jpg"  # Replace with your image path
    output_csv = "extracted_table.csv"  # Output CSV path
    
    df = extract_table_from_image(image_path, output_csv)
    if df is not None:
        print("\nExtracted Data Preview:")
        print(df.head())
