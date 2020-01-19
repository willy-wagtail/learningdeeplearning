def image2vector(image):
    """
    Typically images will be (num_px_x, num_px_y, 3) where 3 represents the RGB values

    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length * height * depth, 1)
    """

    image_shape = image.shape
    v = image.reshape(image_shape[0] * image_shape[1] * image_shape[2], 1)
    return v
