def crop_image(image, x, y, width, height):
    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)
    return image[y:y+height, x:x+width]