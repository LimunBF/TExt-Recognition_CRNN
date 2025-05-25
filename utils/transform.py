from PIL import Image, ImageOps

class ResizeWithPadding:
    def __init__(self, size):
        self.size = size  # (H, W)

    def __call__(self, image):
        image = ImageOps.grayscale(image)
        return ImageOps.pad(image, self.size, color=255, method=Image.BILINEAR)
