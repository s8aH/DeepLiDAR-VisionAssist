import os
import argparse
from glob import iglob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def get_common_basenames(directory, ext_map):
    """Find common basenames across all subdirectories"""
    base_counts = {}
    for dir_name, ext in ext_map.items():
        dir_path = os.path.join(directory, dir_name)
        if not os.path.exists(dir_path):
            continue
            
        files = sorted(iglob(os.path.join(dir_path, f'*.{ext}')))
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0]
            base_counts[base] = base_counts.get(base, 0) + 1

    # Only keep basenames present in ALL directories
    required_count = len([d for d in ext_map if os.path.exists(os.path.join(directory, d))])
    common_bases = [k for k, v in base_counts.items() if v == required_count]
    common_bases.sort()
    return common_bases

def rename_files(directory, ext_map):
    """Rename files with consistent indices across all types"""
    dir_type = os.path.basename(directory)
    common_bases = get_common_basenames(directory, ext_map)
    
    if not common_bases:
        print(f"No common files found in {directory}")
        return

    # Create mapping from old base to new index
    rename_map = {}
    for new_idx, old_base in enumerate(common_bases):
        rename_map[old_base] = f"{new_idx:06d}"

    # Process each directory
    for dir_name, ext in ext_map.items():
        dir_path = os.path.join(directory, dir_name)
        if not os.path.exists(dir_path):
            continue

        files = sorted(iglob(os.path.join(dir_path, f'*.{ext}')))
        if not files:
            continue

        # Prepare rename operations
        operations = []
        for f in files:
            old_base = os.path.splitext(os.path.basename(f))[0]
            if old_base in rename_map:
                new_path = os.path.join(dir_path, f"{rename_map[old_base]}.{ext}")
                operations.append((f, new_path))

        # Execute renames in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(
                executor.map(lambda x: os.rename(*x), operations),
                total=len(operations),
                desc=f"{dir_type}/{dir_name}"
            ))

def convert_kitti_filenames(root_dir, mode='both'):
    ext_map = {
        'velodyne': 'bin',
        'calib': 'txt',
        'label_2': 'txt',
        'image_2': 'png'
    }

    if mode == 'both':
        directories = ['training', 'testing']
    else:
        directories = [mode]

    for subdir in directories:
        dir_path = os.path.join(root_dir, subdir)
        if os.path.exists(dir_path):
            rename_files(dir_path, ext_map)
        else:
            print(f"Directory not found: {dir_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, 
                       help='Path to kitti_format directory')
    parser.add_argument('--mode', choices=['training', 'testing', 'both'], 
                       default='both',
                       help='Which directories to process')
    args = parser.parse_args()

    convert_kitti_filenames(args.data_root, args.mode)