# this code takes images from the unprocessed folders and randomly splits them into their groups
# within the training and test data.
# faster than doing it manually.
# basically when you take all your photos, place all the overhydrated photos into the 
# unprocessed_overhydrated folder, and likewise for the other two groups. Then run this file
# which will split them and move them to their new folders.
import os
import random
import shutil
from pathlib import Path

# ensures no duplicates or overwrites
def move_file_safe(src, dest_dir):
    dest_path = dest_dir / src.name
    counter = 1
    while dest_path.exists():
        new_name = f"{src.stem}_{counter}{src.suffix}"
        dest_path = dest_dir / new_name
        counter += 1
    shutil.move(str(src), dest_path)

# you can change the percentages here if you want a different ratio of splitting the images
def split_data(source_dir, train_dir, valid_dir, test_dir, train_pct=0.7, valid_pct=0.2, test_pct=0.1):
    # stops if wrong values are entered
    assert train_pct + valid_pct + test_pct >= 0.999, "Percentages must add up to 1"
    
    # paths
    all_files = [f for f in Path(source_dir).glob('*') if f.is_file()]
    total_files = len(all_files)
    
    # randomize the files and split them up
    random.shuffle(all_files)
    train_files = all_files[:int(total_files * train_pct)]
    valid_files = all_files[int(total_files * train_pct):int(total_files * (train_pct + valid_pct))]
    test_files = all_files[int(total_files * (train_pct + valid_pct)):]
    
    # moving the files
    for f in train_files:
        move_file_safe(f, train_dir)
    for f in valid_files:
        move_file_safe(f, valid_dir)
    for f in test_files:
        move_file_safe(f, test_dir)

# change paths and folder names as needed, these are just what i had for my computer
def main():
    base_path = Path("/Users/alextowle/orchids/COM Research Seminar/orchid_data/HERO7 BLACK")
    print("Base path exists:", base_path.exists())  # check for base path - sometimes can't find it? check directory

    # where the program moves the images from
    unprocessed_folders = {
        'overhydrated': base_path / "unprocessed_overhydrated",
        'underhydrated': base_path / "unprocessed_underhydrated",
        'control': base_path / "unprocessed_control"
    }
    
    # where the program moves images to
    target_folders = {
        'train': base_path / "train",
        'valid': base_path / "valid",
        'test': base_path / "test"
    }

    # call split_data for the paths
    for category, source_dir in unprocessed_folders.items():
        train_dir = target_folders['train'] / category
        valid_dir = target_folders['valid'] / category
        test_dir = target_folders['test'] / category
        
        # making sure the directories exist
        train_dir.mkdir(parents=True, exist_ok=True)
        valid_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # split and move the files
        split_data(source_dir, train_dir, valid_dir, test_dir)

if __name__ == "__main__":
    main()