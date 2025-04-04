def optimize_column_widths(binary_image, num_columns=9):
    """
    Optimize column widths by analyzing vertical projection profile
    """
    # Calculate vertical projection profile
    col_projection = np.sum(binary_image, axis=0)
    
    # Visualize the projection
    projection_vis = np.zeros((300, len(col_projection)), dtype=np.uint8)
    for i, val in enumerate(col_projection):
        normalized_val = min(int(val / 10), 299)
        cv2.line(projection_vis, (i, 299), (i, 299-normalized_val), 255, 1)
    cv2.imwrite("column_projection.jpg", projection_vis)
    
    # Find local minima in the projection profile
    column_separators = []
    window_size = len(col_projection) // (num_columns * 2)
    for i in range(window_size, len(col_projection) - window_size):
        left_max = max(col_projection[i-window_size:i])
        right_max = max(col_projection[i+1:i+window_size+1])
        if col_projection[i] < left_max * 0.6 and col_projection[i] < right_max * 0.6:
            column_separators.append(i)
    
    # If we find more separators than needed, select the ones with lowest values
    if len(column_separators) > num_columns - 1:
        # Get the values at separator positions
        separator_values = [(sep, col_projection[sep]) for sep in column_separators]
        # Sort by projection value (ascending)
        separator_values.sort(key=lambda x: x[1])
        # Take the num_columns-1 separators with lowest values
        column_separators = [sep for sep, _ in separator_values[:num_columns-1]]
        column_separators.sort()
    
    # Add start and end boundaries
    column_boundaries = [0] + column_separators + [len(col_projection)]
    
    return column_boundaries
