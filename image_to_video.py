import os
import movis as mv
from PIL import Image, ImageFilter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cupy as cp
import tempfile
import shutil

def preprocess_image(img_path, temp_dir, target_resolution=(1920, 1080)):
    """
    Preprocesses a single image by adding blurred borders to fit the target resolution.
    The image's original aspect ratio is preserved.
    """
    with Image.open(img_path) as img:
        # Ensure the image is in RGB mode
        img = img.convert("RGB")

        # Convert image to NumPy array
        img_np = np.array(img, dtype=np.uint8)

        # Use GPU to create a blurred background
        img_gpu = cp.array(img_np)  # Move image to GPU
        blurred_bg_gpu = cp.array(Image.fromarray(img_np).resize(target_resolution))  # Resize
        blurred_bg_gpu = cp.asarray(
            Image.fromarray(cp.asnumpy(blurred_bg_gpu)).filter(ImageFilter.GaussianBlur(20))  # Blur
        )
        blurred_bg = cp.asnumpy(blurred_bg_gpu)  # Move back to CPU

        # Create processed image with blurred background
        processed_img = Image.new("RGB", target_resolution)
        processed_img.paste(Image.fromarray(blurred_bg), (0, 0))

        # Add the original image, centered
        img.thumbnail(target_resolution, Image.Resampling.LANCZOS)
        x_offset = (target_resolution[0] - img.width) // 2
        y_offset = (target_resolution[1] - img.height) // 2
        processed_img.paste(img, (x_offset, y_offset))

        # Save to temp directory
        temp_path = Path(temp_dir) / f"{Path(img_path).stem}_processed.png"
        processed_img.save(temp_path, "PNG")
        return temp_path

def preprocess_images(image_paths, target_resolution=(1920, 1080)):
    """
    Preprocesses multiple images using concurrency and stores them in a temporary directory.
    """
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory
    processed_images = []

    try:
        with ThreadPoolExecutor() as executor:
            future_to_image = {
                executor.submit(preprocess_image, img_path, temp_dir, target_resolution): img_path
                for img_path in image_paths
            }
            for future in as_completed(future_to_image):
                processed_images.append(future.result())
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        shutil.rmtree(temp_dir)  # Cleanup on error
        raise
    return processed_images, temp_dir

def create_video_from_images(images, output_path, image_duration=10.0, transition_duration=0.5):
    """
    Creates a video using images with transitions and sync.
    """
    scene_duration = len(images) * (image_duration - transition_duration) + transition_duration

    # Create a composition
    scene = mv.layer.Composition(size=(1920, 1080), duration=scene_duration)

    # Add image layers with transitions
    offset = 0.0
    for img_path in images:
        image_layer = mv.layer.Image(img_path)
        layer_item = scene.add_layer(
            image_layer,
            offset=offset,
            position=(1920 // 2, 1080 // 2),
            scale=1.0,
            anchor_point=(0.5, 0.5),
        )
        layer_item.opacity.enable_motion().extend(
            keyframes=[0.0, transition_duration, image_duration - transition_duration, image_duration],
            values=[0.0, 1.0, 1.0, 0.0],
        )
        offset += image_duration - transition_duration

    # Write the video
    scene.write_video(output_path)
    print(f"Video created successfully: {output_path}")

# Example usage
if __name__ == "__main__":
    folder_path = "./images"
    output_video = "final_video.mp4"

    # Get all images from folder
    image_files = [str(p) for p in Path(folder_path).glob("*.png")]

    # Preprocess images with concurrency
    processed_images, temp_dir = preprocess_images(image_files)  # Use preprocess_images here

    try:
        # Create video
        create_video_from_images(processed_images, output_video, image_duration=5.0, transition_duration=0.5)
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"Temporary files cleaned up from: {temp_dir}")
