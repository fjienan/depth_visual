import cv2
import os
class photos_save:
    def __init__(self,path = "./photos"):
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
    def save_photo(self, img, prefix="image"):
        count = len(os.listdir(self.path))
        filename = f"{prefix}_{count:04d}.png"
        filepath = os.path.join(self.path, filename)
        cv2.imwrite(filepath, img)
        print(f"Saved photo: {filepath}")
        
        