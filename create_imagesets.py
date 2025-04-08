import os
import random

def generate_imagesets(data_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Generate ImageSets for KITTI-format dataset
    Args:
        data_dir: Path to dataset (containing 'training' folder)
        output_dir: Where to save ImageSets
        train_ratio: Ratio for training set (default: 0.8)
        val_ratio: Ratio for validation set (default: 0.1)
        # test_ratio is automatically calculated as 1 - train_ratio - val_ratio
    """
    # Create output directory
    os.makedirs(os.path.join(output_dir, 'ImageSets'), exist_ok=True)
    
    # Get all sample names (from velodyne folder)
    velodyne_dir = os.path.join(data_dir, 'training', 'velodyne')
    samples = [f.split('.')[0] for f in os.listdir(velodyne_dir) if f.endswith('.bin')]
    samples.sort()  # Sort for consistency
    
    # Shuffle samples randomly
    random.shuffle(samples)
    
    # Calculate split sizes
    total = len(samples)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    
    # Split samples
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    # Write to files
    def write_list_to_file(filepath, sample_list):
        with open(filepath, 'w') as f:
            f.write('\n'.join(sample_list))
    
    write_list_to_file(os.path.join(output_dir, 'ImageSets', 'train.txt'), train_samples)
    write_list_to_file(os.path.join(output_dir, 'ImageSets', 'val.txt'), val_samples)
    write_list_to_file(os.path.join(output_dir, 'ImageSets', 'trainval.txt'), train_samples + val_samples)
    write_list_to_file(os.path.join(output_dir, 'ImageSets', 'test.txt'), test_samples)
    
    print(f"Generated ImageSets with {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples")

# Example usage for your pedestrian-heavy dataset
generate_imagesets(
    data_dir='pointpillars/dataset/kitti_sample',
    output_dir='kitti_sample/ImageSets',
    train_ratio=0.8,
    val_ratio=0.1
)