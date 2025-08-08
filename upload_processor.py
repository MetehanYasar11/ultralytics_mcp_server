#!/usr/bin/env python3
"""
Background dataset upload processor
Bu script Streamlit'den bağımsız olarak çalışır ve büyük dosyaları güvenle işler
"""

import os
import sys
import json
import zipfile
import shutil
import yaml
import time
from pathlib import Path
import tempfile
import signal

def signal_handler(sig, frame):
    print('Upload interrupted by user')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def process_dataset_upload(upload_file_path, dataset_name, status_file):
    """Process dataset upload with intelligent ZIP structure detection"""
    try:
        update_status(status_file, "processing", "Analyzing ZIP structure...")
        
        # Create directories
        custom_datasets_dir = Path("/ultralytics/custom_datasets")
        custom_datasets_dir.mkdir(exist_ok=True)
        
        dataset_dir = custom_datasets_dir / dataset_name
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir()
        
        # Create a temporary extraction directory
        temp_extract_dir = Path("/tmp") / f"extract_{dataset_name}_{int(time.time())}"
        temp_extract_dir.mkdir(exist_ok=True)
        
        try:
            update_status(status_file, "processing", "Extracting ZIP file...")
            
            # Extract entire ZIP to temp directory first
            with zipfile.ZipFile(upload_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
            
            update_status(status_file, "processing", "Analyzing dataset structure...")
            
            # Find data.yaml file location to determine structure
            data_yaml_files = list(temp_extract_dir.rglob("data.yaml")) + list(temp_extract_dir.rglob("data.yml"))
            
            if not data_yaml_files:
                update_status(status_file, "error", "No data.yaml file found in ZIP")
                return False
            
            # Use the first data.yaml found
            data_yaml_path = data_yaml_files[0]
            dataset_root = data_yaml_path.parent
            
            update_status(status_file, "processing", f"Found dataset root at: {dataset_root.name}")
            
            # Look for images and labels directories relative to data.yaml
            images_dirs = list(dataset_root.rglob("images"))
            labels_dirs = list(dataset_root.rglob("labels"))
            
            if not images_dirs:
                update_status(status_file, "error", "No 'images' directory found relative to data.yaml")
                return False
                
            if not labels_dirs:
                update_status(status_file, "error", "No 'labels' directory found relative to data.yaml")
                return False
            
            update_status(status_file, "processing", "Organizing dataset structure...")
            
            # Copy the organized structure to final dataset directory
            files_copied = 0
            
            # Copy data.yaml
            shutil.copy2(data_yaml_path, dataset_dir / "data.yaml")
            files_copied += 1
            
            # Copy images directory
            images_source = images_dirs[0]
            images_dest = dataset_dir / "images"
            if images_source.exists():
                shutil.copytree(images_source, images_dest)
                image_files = len(list(images_dest.rglob("*.*")))
                files_copied += image_files
                update_status(status_file, "processing", f"Copied {image_files} image files")
            
            # Copy labels directory
            labels_source = labels_dirs[0]
            labels_dest = dataset_dir / "labels"
            if labels_source.exists():
                shutil.copytree(labels_source, labels_dest)
                label_files = len(list(labels_dest.rglob("*.*")))
                files_copied += label_files
                update_status(status_file, "processing", f"Copied {label_files} label files")
            
            # Copy any additional directories (like val, test if they exist separately)
            for item in dataset_root.iterdir():
                if item.is_dir() and item.name not in ['images', 'labels']:
                    dest_path = dataset_dir / item.name
                    if not dest_path.exists():
                        try:
                            shutil.copytree(item, dest_path)
                            additional_files = len(list(dest_path.rglob("*.*")))
                            files_copied += additional_files
                            update_status(status_file, "processing", f"Copied additional directory: {item.name} ({additional_files} files)")
                        except Exception as e:
                            print(f"Warning: Could not copy {item.name}: {e}")
            
            update_status(status_file, "processing", "Updating dataset configuration...")
            
            # Update data.yaml paths to be correct
            final_data_yaml = dataset_dir / "data.yaml"
            try:
                with open(final_data_yaml, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # Update paths to be relative to dataset directory
                dataset_path = str(dataset_dir)
                
                # Standard YOLO paths
                if 'train' in config:
                    if not os.path.isabs(config['train']):
                        config['train'] = os.path.join(dataset_path, config['train'])
                    else:
                        # If absolute, make it relative to our dataset
                        config['train'] = os.path.join(dataset_path, "images")
                        
                if 'val' in config:
                    if not os.path.isabs(config['val']):
                        config['val'] = os.path.join(dataset_path, config['val'])
                    else:
                        # Common fallback for validation
                        val_dir = dataset_dir / "val" / "images"
                        if val_dir.exists():
                            config['val'] = str(val_dir)
                        else:
                            config['val'] = os.path.join(dataset_path, "images")
                
                if 'test' in config:
                    if not os.path.isabs(config['test']):
                        config['test'] = os.path.join(dataset_path, config['test'])
                    else:
                        test_dir = dataset_dir / "test" / "images"
                        if test_dir.exists():
                            config['test'] = str(test_dir)
                        else:
                            config['test'] = os.path.join(dataset_path, "images")
                
                # If no train/val specified, set defaults
                if 'train' not in config:
                    config['train'] = os.path.join(dataset_path, "images")
                if 'val' not in config:
                    config['val'] = os.path.join(dataset_path, "images")
                
                # Write updated config
                with open(final_data_yaml, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False)
                    
                update_status(status_file, "processing", "Dataset configuration updated successfully")
                
            except Exception as e:
                update_status(status_file, "processing", f"Warning: Could not update data.yaml: {e}")
            
            # Verify final structure
            final_images = dataset_dir / "images"
            final_labels = dataset_dir / "labels"
            final_yaml = dataset_dir / "data.yaml"
            
            if not all([final_images.exists(), final_labels.exists(), final_yaml.exists()]):
                missing = []
                if not final_images.exists(): missing.append("images/")
                if not final_labels.exists(): missing.append("labels/")
                if not final_yaml.exists(): missing.append("data.yaml")
                update_status(status_file, "error", f"Final dataset missing: {', '.join(missing)}")
                return False
            
            update_status(status_file, "completed", f"Dataset '{dataset_name}' uploaded successfully ({files_copied} files total)")
            return True
            
        finally:
            # Clean up temp extraction directory
            if temp_extract_dir.exists():
                try:
                    shutil.rmtree(temp_extract_dir)
                except Exception as e:
                    print(f"Warning: Could not cleanup temp directory: {e}")
        
    except Exception as e:
        update_status(status_file, "error", f"Upload failed: {str(e)}")
        return False

def update_status(status_file, status, message):
    """Update status file"""
    status_data = {
        "status": status,
        "message": message,
        "timestamp": time.time()
    }
    
    try:
        with open(status_file, 'w') as f:
            json.dump(status_data, f)
    except:
        pass

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python upload_processor.py <upload_file_path> <dataset_name> <status_file>")
        sys.exit(1)
    
    upload_file_path = sys.argv[1]
    dataset_name = sys.argv[2]
    status_file = sys.argv[3]
    
    success = process_dataset_upload(upload_file_path, dataset_name, status_file)
    sys.exit(0 if success else 1)
