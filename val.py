import os
import random
import shutil

# ------------------------------
# CONFIGURATION
# ------------------------------
dataset_root = "dataset"
train_dir = os.path.join(dataset_root, "training")
val_dir = os.path.join(dataset_root, "validation")

images_per_class = 160
random_seed = 42

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# ------------------------------
random.seed(random_seed)

# Get class folders
class_folders = [f for f in os.listdir(train_dir) 
                 if os.path.isdir(os.path.join(train_dir, f))]

os.makedirs(val_dir, exist_ok=True)

total_moved = 0

for cls in class_folders:
    src_class = os.path.join(train_dir, cls)
    dst_class = os.path.join(val_dir, cls)
    
    # List all images in source class
    images = [f for f in os.listdir(src_class) 
              if f.lower().endswith(image_extensions)]
    
    if len(images) < images_per_class:
        print(f"Warning: {cls} has only {len(images)} images, moving all.")
        to_move = images
    else:
        to_move = random.sample(images, images_per_class)
    
    # Create destination folder
    os.makedirs(dst_class, exist_ok=True)
    
    # Move each file
    for img in to_move:
        src_path = os.path.join(src_class, img)
        dst_path = os.path.join(dst_class, img)
        
        # Avoid name collisions (just in case)
        if os.path.exists(dst_path):
            base, ext = os.path.splitext(img)
            counter = 1
            while os.path.exists(os.path.join(dst_class, f"{base}_{counter}{ext}")):
                counter += 1
            dst_path = os.path.join(dst_class, f"{base}_{counter}{ext}")
        
        shutil.move(src_path, dst_path)   # <-- This REMOVES from training
        total_moved += 1
    
    print(f"Moved {len(to_move)} images from {cls} → validation")

print(f"\nTotal moved: {total_moved} (expected {len(class_folders) * images_per_class})")