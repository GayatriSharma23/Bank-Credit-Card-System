def detect_table_lines(img):
    """
    Detect horizontal and vertical lines in a table with improved filtering
    """
    # Enhanced binary image
    binary = enhance_table_image(img)
    
    # Detect horizontal lines - increase kernel size for better line detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Detect vertical lines - increase kernel size to reduce false detections
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    
    # Save for debugging
    cv2.imwrite("horizontal_lines.jpg", horizontal_lines)
    cv2.imwrite("vertical_lines.jpg", vertical_lines)
    
    # Combine lines
    table_grid = cv2.add(horizontal_lines, vertical_lines)
    cv2.imwrite("table_grid.jpg", table_grid)
    
    return table_grid, horizontal_lines, vertical_lines

def filter_line_segments(segments, min_distance=20):
    """
    Filter line segments that are too close to each other
    """
    if not segments:
        return []
    
    filtered = [segments[0]]
    for i in range(1, len(segments)):
        # Check if this segment is far enough from the last accepted segment
        if segments[i] - filtered[-1] >= min_distance:
            filtered.append(segments[i])
    
    return filtered

def extract_table_from_image(image_path, output_csv=None):
    """
    Main function to extract table from a rotated image with improved filtering
    """
    print(f"Processing image: {image_path}")
    
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'])
    print("EasyOCR initialized")
    
    # Deskew the image
    rotated_img = deskew_image(image_path)
    
    try:
        # Detect table grid
        grid, h_lines, v_lines = detect_table_lines(rotated_img)
        
        # Find row and column boundaries
        row_boundaries = find_horizontal_segments(h_lines)
        col_boundaries = find_vertical_segments(v_lines)
        
        # Filter boundaries that are too close to each other
        filtered_row_boundaries = filter_line_segments(row_boundaries, min_distance=15)
        filtered_col_boundaries = filter_line_segments(col_boundaries, min_distance=15)
        
        print(f"Detected {len(row_boundaries)} rows and {len(col_boundaries)} columns")
        print(f"After filtering: {len(filtered_row_boundaries)} rows and {len(filtered_col_boundaries)} columns")
        
        # Check if we have a reasonable number of columns (9 for your case)
        # If not, try to force 9 columns by selecting the most prominent ones
        expected_columns = 9
        if len(filtered_col_boundaries) > expected_columns + 1:  # +1 because boundaries include start and end
            print(f"Too many columns detected. Forcing {expected_columns} columns.")
            # Get image width
            img_width = rotated_img.shape[1]
            # Create equidistant column boundaries
            filtered_col_boundaries = [int(i * img_width / expected_columns) for i in range(expected_columns + 1)]
        
        if len(filtered_row_boundaries) < 2 or len(filtered_col_boundaries) < 2:
            print("Insufficient grid lines detected, trying direct cell extraction")
            cells = direct_cell_extraction(rotated_img)
        else:
            # Extract cells using filtered grid
            cells = extract_cells(rotated_img, filtered_row_boundaries, filtered_col_boundaries)
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
    
    # Filter out empty rows (rows where all cells are empty)
    table_data = [row for row in table_data if any(cell.strip() for cell in row)]
    
    # Create DataFrame
    if table_data:
        # For your specific case with 9 columns
        expected_headers = ["Sl. No", "State", "District", "Block", "Gram Panchayat", "Village", "Hamlet", "Habitation", "Remarks"]
        
        # Get max columns from non-empty rows
        max_cols = max(len(row) for row in table_data)
        
        # Use expected headers if they fit
        if len(expected_headers) <= max_cols:
            column_names = expected_headers + [f"Column {i+1}" for i in range(len(expected_headers), max_cols)]
        else:
            column_names = expected_headers[:max_cols]
        
        # Ensure all rows have the same number of columns
        for i in range(len(table_data)):
            while len(table_data[i]) < max_cols:
                table_data[i].append("")
            if len(table_data[i]) > max_cols:
                table_data[i] = table_data[i][:max_cols]
        
        # Create DataFrame
        df = pd.DataFrame(table_data, columns=column_names)
        
        # Additional cleaning: Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove rows where all cells contain empty string
        df = df[df.apply(lambda x: x.astype(str).str.strip().str.len() > 0).any(axis=1)]
        
        # Save to CSV
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Table data saved to {output_csv}")
        
        return df
    else:
        print("No valid table data extracted")
        return None
