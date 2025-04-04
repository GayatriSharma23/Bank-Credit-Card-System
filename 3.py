def extract_table_from_image(image_path, output_csv=None):
    """
    More reliable approach using direct grid-based cell extraction
    with proper image handling
    """
    print(f"Processing image: {image_path}")
    
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'])
    print("EasyOCR initialized")
    
    # Load image - ensure proper reading
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Explicitly convert image to proper format
    img = np.array(img, dtype=np.uint8)
    
    # Rotate 90 degrees if needed
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite("rotated_image.jpg", rotated)
    
    # Convert to grayscale
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    
    # Apply preprocessing to enhance table structure
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Filter out small noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cv2.imwrite("binary_table.jpg", binary)
    
    # Get image dimensions
    height, width = binary.shape[:2]
    
    # Define fixed grid for 9 columns
    num_columns = 9
    column_width = width // num_columns
    
    # Count number of rows using horizontal projection profile
    row_projection = np.sum(binary, axis=1).astype(np.uint8)
    row_projection_vis = np.zeros((height, 300), dtype=np.uint8)
    for i, val in enumerate(row_projection):
        normalized_val = min(val // 20, 299)
        cv2.line(row_projection_vis, (0, i), (int(normalized_val), i), 255, 1)
    cv2.imwrite("row_projection.jpg", row_projection_vis)
    
    # Find row boundaries using the projection profile
    row_boundaries = []
    in_row = False
    threshold = width * 0.05  # 5% of width as threshold for detecting rows
    
    for i, projection in enumerate(row_projection):
        if not in_row and projection > threshold:
            # Start of a row
            row_start = i
            in_row = True
        elif in_row and (projection <= threshold or i == height - 1):
            # End of a row or end of image
            row_end = i
            if row_end - row_start > 10:  # Minimum row height
                row_boundaries.append((row_start, row_end))
            in_row = False
    
    print(f"Detected {len(row_boundaries)} rows")
    
    # Draw row boundaries on a copy for visualization
    row_vis = rotated.copy()
    for start, end in row_boundaries:
        cv2.line(row_vis, (0, start), (width, start), (0, 255, 0), 2)
        cv2.line(row_vis, (0, end), (width, end), (0, 0, 255), 2)
    cv2.imwrite("row_boundaries.jpg", row_vis)
    
    # Define expected column headers
    expected_headers = ["Sl. No", "State", "District", "Block", "Gram Panchayat", "Village", "Hamlet", "Habitation", "Remarks"]
    
    # Extract text from each cell in the grid
    table_data = []
    
    for row_idx, (row_start, row_end) in enumerate(row_boundaries):
        row_data = []
        for col_idx in range(num_columns):
            # Define cell boundaries
            col_start = col_idx * column_width
            col_end = (col_idx + 1) * column_width
            
            # Extract cell image - ensure proper array handling
            cell_img = rotated[row_start:row_end, col_start:col_end].copy()
            
            # Save some cells for debugging
            if row_idx < 3 and col_idx < 3:
                cv2.imwrite(f"cell_{row_idx}_{col_idx}.jpg", cell_img)
            
            # Extract text using EasyOCR
            try:
                results = reader.readtext(cell_img)
                cell_text = ' '.join([result[1] for result in results])
                row_data.append(cell_text.strip())
            except Exception as e:
                print(f"Error in OCR for cell ({row_idx}, {col_idx}): {e}")
                row_data.append("")
        
        # Only add rows with some content
        if any(cell.strip() for cell in row_data):
            table_data.append(row_data)
    
    # Create DataFrame
    if table_data:
        # Create DataFrame with fixed columns
        df = pd.DataFrame(table_data, columns=expected_headers)
        
        # Clean the data
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        
        # Remove rows where all entries are empty or very short
        df = df[df.astype(str).apply(lambda x: x.str.strip().str.len() > 2).any(axis=1)]
        
        # Save to CSV
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Table data saved to {output_csv}")
        
        return df
    else:
        print("No valid table data extracted")
        return None
