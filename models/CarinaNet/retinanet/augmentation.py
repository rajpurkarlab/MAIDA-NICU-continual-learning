import math
import os
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
from PIL import Image, ImageDraw

PRINT_IMAGES = False
print(f"PRINT_IMAGES={PRINT_IMAGES}")

def visualize_boxes(image, boxes, save_path="./test", filename="image.png"):
    """
    Draw bounding boxes on the image and save it.
    
    Args:
        image (torch.Tensor): Image tensor of shape [1, C, H, W] or [C, H, W]
        boxes (torch.Tensor): Bounding boxes tensor of shape [N, 5] where each row is [class_id, x1, y1, x2, y2]
        save_path (str): Directory to save the output image
        filename (str): Name of the output file
    """
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # If image is batched, take first image
    if image.dim() == 4:
        image = image[0]
    
    # Convert tensor to PIL Image
    image_pil = F.to_pil_image(image)
    
    # Create drawing object
    draw = ImageDraw.Draw(image_pil)

    # Draw each box
    for box in boxes[0]:  
        # Extract coordinates (box is in [x1, y1, x2, y2, class_id] format)
        class_id = int(box[4].item())

        x1, y1, x2, y2 = box[:4].tolist()
        
        # Draw rectangle
        try:
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline="red",
                width=2
            )
            # Draw the center of the box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            draw.ellipse(
                [center, (center[0] + 5, center[1] + 5)],
                fill="red"
            )
            
            # Optionally, draw class ID
            draw.text((center[0], center[1] - 10), f"Class {class_id}", fill="red")
        except:
            breakpoint()
    
    # Save the image
    save_path = os.path.join(save_path, filename)
    image_pil.save(save_path)
    
    return image_pil

class MedicalImageTransforms:
    def __init__(
        self,
        size=(640, 640),  # Current dataset size
        degrees=10,  # Small rotation range
        p_transform=1,
        crop_percent = 0.25
    ):
        self.size = size
        self.degrees = degrees
        self.p_transform = p_transform
        self.crop_percent = crop_percent
        
        print(f"Initialized MedicalImageTransforms with size={size}, degrees={degrees}, p_transform={p_transform}, crop_percent={crop_percent}")

    def _apply_crop(self, image, boxes):
        """Apply random crop from top of image"""
        _, height, width = image.shape
        
        # Random crop percentage (0 to 15%% from top or botton)
        crop_percent = random.uniform(-self.crop_percent, self.crop_percent)
        
        if crop_percent > 0:
            crop_from_top = True
        else:
            crop_from_top = False
            crop_percent = abs(crop_percent) 
        crop_height = int(height * crop_percent)
        
        if crop_height == 0:
            return image, boxes
        
        # Apply crop
        if crop_from_top:
            image = image[:, crop_height:, :]
        else:
            image = image[:, :-crop_height, :]
        
        if boxes is not None and len(boxes):
            # Adjust box coordinates
            boxes_cropped = boxes.clone()
            
            if crop_from_top:
                # Adjust y coordinates (corner format: [x1, y1, x2, y2])
                boxes_cropped[..., 1] -= crop_height  # y1
                boxes_cropped[..., 3] -= crop_height  # y2
                
            # Scale coordinates to new height
            scale_factor = height / (height - crop_height)
            boxes_cropped[..., 1] *= scale_factor  # y1
            boxes_cropped[..., 3] *= scale_factor  # y2  
            
            # Remove boxes that are now outside the image
            valid_boxes = (boxes_cropped[..., 1] > 0) & (boxes_cropped[..., 3] < height)

            boxes = boxes_cropped[valid_boxes]
            
        # Resize back to original dimensions
        image = F.resize(image, self.size, antialias=True)
     
        
        return image, boxes

    def _apply_rotation(self, image, boxes):
        """Apply random rotation to image and boxes"""
        angle = random.uniform(-self.degrees, self.degrees)
        
        # Get image dimensions
        _, height, width = image.shape

        # Rotate image using torchvision's function
        image = F.rotate(image, angle, fill=0)
        
        if boxes is not None and len(boxes):
            # Convert angle to radians (positive for counter-clockwise)
            angle_rad = - angle * math.pi / 180
            cos_theta = torch.cos(torch.tensor(angle_rad))
            sin_theta = torch.sin(torch.tensor(angle_rad))
            cx = width / 2
            cy = height / 2
                  
            # Rotate box corners
            boxes_rotated = boxes.clone()
            for i in range(len(boxes)):
                # Get corners
                x1, y1 = boxes[i, 0], boxes[i, 1]
                x2, y2 = boxes[i, 2], boxes[i, 3]
                x_mid, y_mid = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Translate to origin (relative to center)
                x1, y1 = x1 - cx, y1 - cy
                x2, y2 = x2 - cx, y2 - cy
                x_mid, y_mid = x_mid - cx, y_mid - cy
                
                # Rotate (counter-clockwise, matching torchvision)
                rx1 = cos_theta * x1 - sin_theta * y1
                ry1 = sin_theta * x1 + cos_theta * y1
                rx2 = cos_theta * x2 - sin_theta * y2
                ry2 = sin_theta * x2 + cos_theta * y2
                r_mid_x = cos_theta * x_mid - sin_theta * y_mid
                r_mid_y = sin_theta * x_mid + cos_theta * y_mid
                
                # Translate back
                boxes_rotated[i, 0] = rx1 + cx  # x1
                boxes_rotated[i, 1] = ry1 + cy  # y1
                boxes_rotated[i, 2] = rx2 + cx  # x2
                boxes_rotated[i, 3] = ry2 + cy  # y2
                r_mid_x_rotated = r_mid_x + cx
                r_mid_y_rotated = r_mid_y + cy
                
                assert (r_mid_x_rotated - (boxes_rotated[i, 0] + boxes_rotated[i, 2]) / 2).abs() < 1e-5
                assert (r_mid_y_rotated - (boxes_rotated[i, 1] + boxes_rotated[i, 3]) / 2).abs() < 1e-5
            
            # Filter out invalid boxes
            valid_boxes = (boxes_rotated[..., 0] > 0) & (boxes_rotated[..., 2] < width) & \
                        (boxes_rotated[..., 1] > 0) & (boxes_rotated[..., 3] < height)
            boxes = boxes_rotated[valid_boxes]
        
        return image, boxes

    def transform_image_boxes(self, image, box):
        image_aug, boxes_aug = self._apply_rotation(image, box)
        image_aug, boxes_aug = self._apply_crop(image_aug, boxes_aug)
        return image_aug,boxes_aug
    
    def __call__(self, images, boxes=None):
        if boxes is None:
            return images

        # Get the original image size
        _, _, height, width = images.shape
        assert height == self.size[0] and width == self.size[1], "Image size mismatch"

    
        # Draw boxes on image and save it to ./test
        if PRINT_IMAGES:
            save_path="./test"
            # Get the number of files in save_path
            num_files = len(os.listdir(save_path))
            visualize_boxes(images, boxes, save_path, filename=f"{num_files}_original_image.png")

        image_augs = []
        boxes_augs = []
        for i, (image, box) in enumerate(zip(images, boxes)):
            if torch.rand(1) <= self.p_transform:   

                image_aug, boxes_aug = self.transform_image_boxes(image, box)
                    
                while len(boxes_aug) != box.shape[0]:
                    image_aug, boxes_aug = self.transform_image_boxes(image, box)
                   
                image_augs.append(image_aug)
                boxes_augs.append(boxes_aug)
        
        assert len(image_augs) == len(boxes_augs), "Number of images and boxes do not match"
        assert len(image_augs) == images.shape[0], "Number of images do not match input batch size"
        assert len(boxes_augs) == len(boxes), "Number of boxes do not match input batch size"

        images = torch.stack(image_augs)
        boxes = boxes_augs

        if PRINT_IMAGES:
            visualize_boxes(images, boxes, save_path, filename=f"{num_files}_transformed.png")

        
        return images, boxes