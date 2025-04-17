

def create_filled_brain_mask(x, threshold_value=0.01):
    """
    Create a binary mask of the brain with all internal structures filled.
    Uses a simple approach with thresholding and 3D hole filling.
    
    Args:
        x (torch.Tensor): Input tensor of shape [batch, H, W, D]
        threshold_value (float): Threshold value to identify brain tissue
        
    Returns:
        torch.Tensor: Binary mask of the same shape as input, with 1s inside the brain region and 0s outside
    """
    # Convert to numpy for processing if needed
    if x.is_cuda:
        x_np = x.cpu().numpy()
    else:
        x_np = x.numpy()
        
    # Get the shape
    batch_size = x_np.shape[0]
    
    # Initialize the result tensor
    result = torch.zeros_like(x)
    
    # Process each element in the batch
    for b in range(batch_size):
        # Create initial binary mask using thresholding
        binary_mask = (x_np[b] > threshold_value).astype(np.int32)
        
        # Label connected components
        labeled_array, num_features = ndimage.label(binary_mask)
        
        if num_features > 0:
            # Find the largest connected component (brain)
            sizes = ndimage.sum(np.ones_like(labeled_array), labeled_array, range(1, num_features + 1))
            largest_component_label = np.argmax(sizes) + 1
            
            # Keep only the largest component (brain)
            brain_mask = (labeled_array == largest_component_label).astype(np.int32)
            
            # Fill holes in the 3D brain mask
            filled_mask = ndimage.binary_fill_holes(brain_mask).astype(np.int32)
            
            # Convert back to tensor
            result[b] = torch.from_numpy(filled_mask).to(x.device)
    
    return result.bool()



def get_background_value(image):
    # Get values from the corners of the image
    corner_size = 5
    corners = [
        image[:corner_size, :corner_size],  # Top-left
        image[:corner_size, -corner_size:],  # Top-right
        image[-corner_size:, :corner_size],  # Bottom-left
        image[-corner_size:, -corner_size:]  # Bottom-right
    ]
    background_samples = np.concatenate([c.flatten() for c in corners])
    # Take the median of corner values
    background_value = np.median(background_samples)
    return background_value