import tarfile
import os
import shutil

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Define the relative path to the file
relative_path = 'datasets/101_ObjectCategories.tar.gz'

# Combine the script directory with the relative path to get the absolute path
tar_file_path = os.path.join(script_dir, relative_path)

extract_dir = 'datasets'

# Extract the archive
with tarfile.open(tar_file_path, 'r:gz') as tar:
    tar.extractall(extract_dir)

# Specify the path to the directory we want to remove
directory_to_remove = 'datasets/caltech101/BACKGROUND_Google'

try:
    # Remove the directory
    shutil.rmtree(directory_to_remove)
    print(f"Directory '{directory_to_remove}' has been successfully removed.")
except FileNotFoundError:
    print(f"Directory '{directory_to_remove}' does not exist.")
except Exception as e:
    print(f"An error occurred while removing the directory: {e}")