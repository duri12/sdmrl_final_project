import os
import re

folder_path = '../checkpoints'

for filename in os.listdir(folder_path):
    match = re.fullmatch(r'msac_step(\d{6})\.zip', filename)

    if match:
        # Extract the numeric part
        number = match.group(1)
        new_name = f'msac_step_{number}.zip'

        # Build full paths
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {new_name}')