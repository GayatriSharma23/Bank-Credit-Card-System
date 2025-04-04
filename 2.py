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

def deskew_image(image_path):
    """
    Fix the rotation of a heavily skewed image
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use HoughLines to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    
    if lines is None:
        print("No lines detected for rotation calculation")
        return img
    
    # Find the dominant angle (should be around -45 degrees)
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle_deg = np.degrees(theta) - 90
        # Normalize angle
        if angle_deg < -90:
            angle_deg += 180
        elif angle_deg > 90:
            angle_deg -= 180
        angles.append(angle_deg)
    
    # Calculate the average angle
    angle = np.median(angles)
    print(f"Detected angle: {angle} degrees")
    
    # Explicitly force angle to be -45 if detection is close
    if -55 < angle < -35:
        angle = -45
        print(f"Using fixed rotation angle: {angle} degrees")
    
    # Get image dimensions
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    
    # Calculate rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)
    
    # Adjust rotation matrix
    M[0, 2] += new_width / 2 - center[0]
    M[1, 2] += new_height / 2 - center[1]
    
    # Perform rotation
    rotated = cv2.warpAffine(img, M, (new_width, new_height), 
                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, 
                           borderValue=(255, 255, 255))
    
    # Save the rotated image for verification
    cv2.imwrite("rotated_image.jpg", rotated)
    
    return rotated

def enhance_table_image(img):
    """
    Enhance the table image for better line detection
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Dilate to strengthen lines
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Save enhanced image
    cv2.imwrite("enhanced_table.jpg", dilated)
    
    return dilated

def detect_table_lines(img):
    """
    Detect horizontal and vertical lines in a table
    """
    # Enhanced binary image
    binary = enhance_table_image(img)
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    cv2.imwrite("horizontal_lines.jpg", horizontal_lines)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    cv2.imwrite("vertical_lines.jpg", vertical_lines)
    
    # Combine lines
    table_grid = cv2.add(horizontal_lines, vertical_lines)
    cv2.imwrite("table_grid.jpg", table_grid)
    
    return table_grid, horizontal_lines, vertical_lines

def find_horizontal_segments(horizontal_lines):
    """
    Detect row segments using horizontal lines
    """
    # Find contours of horizontal lines
    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract y-coordinates of horizontal lines (row boundaries)
    row_boundaries = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Store the middle point of the line
        row_boundaries.append(y + h//2)
    
    # Sort by y-coordinate
    row_boundaries.sort()
    
    return row_boundaries

def find_vertical_segments(vertical_lines):
    """
    Detect column segments using vertical lines
    """
    # Find contours of vertical lines
    contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract x-coordinates of vertical lines (column boundaries)
    col_boundaries = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Store the middle point of the line
        col_boundaries.append(x + w//2)
    
    # Sort by x-coordinate
    col_boundaries.sort()
    
    return col_boundaries

def extract_cells(img, row_boundaries, col_boundaries):
    """
    Extract cells from the image using row and column boundaries
    """
    cells = []
    
    # Convert boundaries to cell regions
    if not row_boundaries or not col_boundaries:
        print("Row or column boundaries not detected")
        return cells
        
    # Create cells from boundaries
    for i in range(len(row_boundaries) - 1):
        row_cells = []
        for j in range(len(col_boundaries) - 1):
            # Cell boundaries
            top = row_boundaries[i]
            bottom = row_boundaries[i + 1]
            left = col_boundaries[j]
            right = col_boundaries[j + 1]
            
            # Extract cell image
            # Add some padding
            padding = 2
            top = max(0, top - padding)
            left = max(0, left - padding)
            bottom = min(img.shape[0], bottom + padding)
            right = min(img.shape[1], right + padding)
            
            cell_img = img[top:bottom, left:right]
            
            # Add to row
            row_cells.append((cell_img, left, top, right-left, bottom-top))
        
        cells.append(row_cells)
    
    return cells

def extract_text_from_cells(cells, reader):
    """
    Extract text from cell images using OCR
    """
    table_data = []
    
    for row_idx, row in enumerate(cells):
        row_data = []
        for cell_idx, (cell_img, x, y, w, h) in enumerate(row):
            # Skip empty cells or cells that are too small
            if cell_img.size == 0 or w < 5 or h < 5:
                row_data.append("")
                continue
            
            # Save some cells for debugging
            if row_idx < 3 and cell_idx < 3:
                cv2.imwrite(f"cell_{row_idx}_{cell_idx}.jpg", cell_img)
            
            # Extract text using EasyOCR
            try:
                results = reader.readtext(cell_img)
                cell_text = ' '.join([result[1] for result in results])
                row_data.append(cell_text.strip())
            except Exception as e:
                print(f"Error in OCR for cell ({row_idx}, {cell_idx}): {e}")
                row_data.append("")
        
        # Only add rows with some content
        if any(cell.strip() for cell in row_data):
            table_data.append(row_data)
    
    return table_data

def direct_cell_extraction(img):
    """
    Alternative approach: directly extract cells using contour detection
    when line detection fails
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations to enhance cell boundaries
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find potential cells
    cell_contours = []
    min_area = 100  # Minimum area for a cell
    max_area = img.shape[0] * img.shape[1] // 10  # Maximum area (1/10 of image)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            # Add some constraints for cell shapes
            if 10 < w < 200 and 10 < h < 100:
                cell_contours.append((x, y, w, h))
    
    # Draw contours on a copy of the image for debugging
    contour_img = img.copy()
    for x, y, w, h in cell_contours:
        cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite("detected_cells.jpg", contour_img)
    
    # Try to organize cells into a grid
    # First sort by y-coordinate to identify rows
    cell_contours.sort(key=lambda c: c[1])
    
    # Group cells by row
    rows = []
    current_row = [cell_contours[0]]
    current_y = cell_contours[0][1]
    
    for cell in cell_contours[1:]:
        y_diff = abs(cell[1] - current_y)
        if y_diff < 20:  # Cells in same row if y difference is small
            current_row.append(cell)
        else:
            # Sort current row by x-coordinate
            current_row.sort(key=lambda c: c[0])
            rows.append(current_row)
            # Start new row
            current_row = [cell]
            current_y = cell[1]
    
    # Add last row
    if current_row:
        current_row.sort(key=lambda c: c[0])
        rows.append(current_row)
    
    # Extract cell images
    cell_images = []
    for row in rows:
        row_images = []
        for x, y, w, h in row:
            cell_img = img[y:y+h, x:x+w]
            row_images.append((cell_img, x, y, w, h))
        cell_images.append(row_images)
    
    return cell_images

def extract_table_from_image(image_path, output_csv=None):
    """
    Main function to extract table from a rotated image
    """
    print(f"Processing image: {image_path}")
    
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'])
    print("EasyOCR initialized")
    
    # Deskew the image
    rotated_img = deskew_image(image_path)
    
    # Try line-based approach first
    try:
        # Detect table grid
        grid, h_lines, v_lines = detect_table_lines(rotated_img)
        
        # Find row and column boundaries
        row_boundaries = find_horizontal_segments(h_lines)
        col_boundaries = find_vertical_segments(v_lines)
        
        print(f"Detected {len(row_boundaries)} rows and {len(col_boundaries)} columns")
        
        if len(row_boundaries) < 2 or len(col_boundaries) < 2:
            print("Insufficient grid lines detected, trying direct cell extraction")
            cells = direct_cell_extraction(rotated_img)
        else:
            # Extract cells using grid
            cells = extract_cells(rotated_img, row_boundaries, col_boundaries)
    except Exception as e:
        print(f"Grid detection failed: {e}")
        print("Falling back to direct cell extraction")
        cells = direct_cell_extraction(rotated_img)
    
    # If no cells were found, report failure
    if not cells:
        print("Failed to detect table structure")
        return None
    
    print(f"Extracted {len(cells)} rows of cells")
    
    # Extract text from cells
    table_data = extract_text_from_cells(cells, reader)
    
    # Create DataFrame
    if table_data:
        # Determine max columns
        max_cols = max(len(row) for row in table_data)
        
        # Create column headers
        # For your specific case, use known column names if possible
        expected_headers = ["Sl. No", "State", "District", "Block", "Gram Panchayat", "Village"]
        if len(expected_headers) <= max_cols:
            # Pad with generic headers if needed
            while len(expected_headers) < max_cols:
                expected_headers.append(f"Column {len(expected_headers) + 1}")
            column_names = expected_headers[:max_cols]
        else:
            # Use generic headers if we have more columns than expected
            column_names = [f"Column {i+1}" for i in range(max_cols)]
        
        # Ensure all rows have the same number of columns
        for i in range(len(table_data)):
            while len(table_data[i]) < max_cols:
                table_data[i].append("")
            # Truncate if longer than max columns
            if len(table_data[i]) > max_cols:
                table_data[i] = table_data[i][:max_cols]
        
        # Create DataFrame
        df = pd.DataFrame(table_data, columns=column_names)
        
        # Clean the data (remove extra spaces, fix OCR errors)
        for col in df.columns:
            if df[col].dtype == 'object':  # Only clean string columns
                df[col] = df[col].str.strip()
                # Replace multiple spaces with a single space
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        
        # Save to CSV
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Table data saved to {output_csv}")
        
        return df
    else:
        print("No valid table data extracted")
        return None

# Example usage
if __name__ == "__main__":
    image_path = "table_image.jpg"  # Replace with your image path
    output_csv = "extracted_table.csv"  # Output CSV path
    
    df = extract_table_from_image(image_path, output_csv)
    if df is not None:
        print("\nExtracted Data Preview:")
        print(df.head())
