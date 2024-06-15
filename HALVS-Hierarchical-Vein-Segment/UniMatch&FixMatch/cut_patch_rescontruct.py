from PIL import Image
import numpy as np
import os

from PIL import Image
import numpy as np
import os

def combine_patches_from_folder(original_image_path, patch_folder, output_image_path):
    # Load the original image to get its size
    original_image = Image.open(original_image_path)
    width, height = original_image.size
    combined = np.zeros((height, width, 3), dtype=np.uint8)

    # Get sorted list of patch filenames
    patch_filenames = sorted(os.listdir(patch_folder), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    patch_size = Image.open(os.path.join(patch_folder, patch_filenames[0])).size[0]

    # Combine patches
    patch_id = 0
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch_path = os.path.join(patch_folder, patch_filenames[patch_id])
            patch = np.array(Image.open(patch_path))

            patch_height, patch_width = patch.shape[0], patch.shape[1]

            combined[i:i + patch_height, j:j + patch_width] = patch

            patch_id += 1

    # Save combined image
    combined_image = Image.fromarray(combined)
    combined_image.save(output_image_path)

def batch_split_image_into_patches_and_save_filenames(image_folder, patch_size=256, output_folder='patches', txt_folder='txt'):
    image_files = os.listdir(image_folder)

    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)
        image = np.array(image)

        # Create output directory if it doesn't exist
        output_dir = os.path.join(output_folder, image_file.split('.')[0])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Open text file for writing patch filenames
        txt_path = os.path.join(txt_folder, f'{image_file.split(".")[0]}.txt')
        with open(txt_path, 'w') as f:
            # Split image into patches and save each patch as a separate image
            patch_id = 100
            for i in range(0, image.shape[0], patch_size):
                for j in range(0, image.shape[1], patch_size):
                    patch = image[i:i + patch_size, j:j + patch_size]
                    patch_image = Image.fromarray(patch)
                    patch_filename = f'{image_file.split(".")[0]}_patch_{patch_id}.png'
                    patch_image.save(os.path.join(output_dir, patch_filename))

                    # Write patch filename to text file in the specified format
                    f.write(f'JPEGImages/{patch_filename} SegmentationClass/{patch_filename}\n')

                    patch_id += 1

def batch_combine_patches_from_folder(original_image_folder, patch_folders, output_folder):
    original_image_files = os.listdir(original_image_folder)
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for original_image_file in original_image_files:
        original_image_path = os.path.join(original_image_folder, original_image_file)
        patch_folder = os.path.join(patch_folders, original_image_file.split('.')[0])
        output_image_path = os.path.join(output_folder, original_image_file)
        combine_patches_from_folder(original_image_path, patch_folder, output_image_path)

# Call the functions
# batch_split_image_into_patches_and_save_filenames('img/root', 256, 'img/patch_mask', 'img')
# After predicting patches, call the function to combine patches
batch_combine_patches_from_folder('img/root', 'img/recover_img', 'img/combined')

