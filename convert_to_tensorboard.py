"""
Convert existing YOLO training CSV results to TensorBoard format
This script creates TensorBoard event files from results.csv for visualization
"""

import pandas as pd
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import yaml
import sys

def get_model_name(training_dir):
    """Extract a short, meaningful model name from args.yaml"""
    args_file = training_dir / "args.yaml"
    
    if args_file.exists():
        try:
            with open(args_file, 'r') as f:
                args = yaml.safe_load(f)
                
                # Get model variant (yolo11n, yolo11s, etc.)
                model_variant = "yolo11"
                if 'model' in args:
                    model_path = str(args['model'])
                    # Extract just the model name without .pt
                    model_name = Path(model_path).stem
                    if model_name.startswith('yolo'):
                        model_variant = model_name
                
                # Get dataset name (simplified)
                dataset_name = "unknown"
                if 'data' in args:
                    data_path = Path(str(args['data']))
                    # Get parent directory name as dataset
                    parent_dir = data_path.parent.name
                    if parent_dir and parent_dir not in ['custom_datasets', 'YOLO_MultiLevel_Datasets']:
                        # Shorten long dataset names
                        if 'kaggle' in parent_dir.lower():
                            dataset_name = "kaggle"
                        elif 'panoramic' in parent_dir.lower():
                            dataset_name = "panoramic"
                        elif 'dental' in parent_dir.lower():
                            dataset_name = "dental"
                        else:
                            # Take first word or first 10 chars
                            words = parent_dir.split()
                            dataset_name = words[0] if words else parent_dir[:10]
                    elif parent_dir == 'YOLO_MultiLevel_Datasets':
                        # For built-in datasets, use the folder name before data.yaml
                        folder_name = data_path.parent.parent.name if data_path.parent.parent.exists() else "builtin"
                        dataset_name = folder_name.replace('YOLO_', '').replace('_', '')
                
                # Get epochs
                epochs = args.get('epochs', '?')
                
                # Create compact name: model_dataset_epochs
                return f"{model_variant}_{dataset_name}_{epochs}ep"
                
        except Exception as e:
            print(f"  Warning: Could not parse args.yaml: {e}")
            pass
    
    # Fallback to training directory name
    return training_dir.name

def convert_csv_to_tensorboard(training_dir, tensorboard_root):
    """Convert a single training's CSV results to TensorBoard format"""
    
    results_csv = training_dir / "results.csv"
    
    if not results_csv.exists():
        print(f"⚠️  No results.csv found in {training_dir.name}")
        return False
    
    try:
        # Get model name
        model_name = get_model_name(training_dir)
        
        # Read CSV results
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # Remove whitespace from column names
        
        # Create TensorBoard writer with model name as the run name
        # This will show as "model_name" in TensorBoard UI
        log_dir = tensorboard_root / model_name
        writer = SummaryWriter(log_dir=str(log_dir))
        
        print(f"📊 Converting {training_dir.name} → {model_name}...")
        
        # Add metadata
        writer.add_text("Training/Directory", training_dir.name, 0)
        writer.add_text("Training/Epochs", str(len(df)), 0)
        
        # Map CSV columns to TensorBoard metrics
        metric_mapping = {
            'train/box_loss': 'train/box_loss',
            'train/cls_loss': 'train/cls_loss', 
            'train/dfl_loss': 'train/dfl_loss',
            'metrics/precision(B)': 'metrics/precision',
            'metrics/recall(B)': 'metrics/recall',
            'metrics/mAP50(B)': 'metrics/mAP50',
            'metrics/mAP50-95(B)': 'metrics/mAP50-95',
            'val/box_loss': 'val/box_loss',
            'val/cls_loss': 'val/cls_loss',
            'val/dfl_loss': 'val/dfl_loss',
            'lr/pg0': 'learning_rate/pg0',
            'lr/pg1': 'learning_rate/pg1',
            'lr/pg2': 'learning_rate/pg2'
        }
        
        # Write metrics to TensorBoard
        for idx, row in df.iterrows():
            epoch = int(row['epoch']) if 'epoch' in df.columns else idx
            
            for csv_col, tb_name in metric_mapping.items():
                if csv_col in df.columns:
                    value = row[csv_col]
                    if pd.notna(value):  # Only write non-NaN values
                        writer.add_scalar(tb_name, float(value), epoch)
        
        writer.close()
        print(f"✅ Successfully converted {training_dir.name} → {model_name} ({len(df)} epochs)")
        return True
        
    except Exception as e:
        print(f"❌ Error converting {training_dir.name}: {str(e)}")
        return False

def main():
    # Find all training directories
    runs_dir = Path("ultralytics/runs/detect")
    
    if not runs_dir.exists():
        print(f"❌ Directory not found: {runs_dir}")
        return
    
    # Create central TensorBoard logs directory
    tensorboard_root = runs_dir / "tensorboard_logs"
    tensorboard_root.mkdir(exist_ok=True)
    
    training_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name != "tensorboard_logs"]
    
    if not training_dirs:
        print("❌ No training directories found!")
        return
    
    print(f"\n🔄 Found {len(training_dirs)} training directories")
    print("=" * 60)
    
    success_count = 0
    total_count = len(training_dirs)
    
    for training_dir in sorted(training_dirs):
        if convert_csv_to_tensorboard(training_dir, tensorboard_root):
            success_count += 1
    
    print("=" * 60)
    print(f"\n✨ Conversion complete: {success_count}/{total_count} successful")
    print(f"\n🚀 Now run: tensorboard --logdir=ultralytics/runs/detect/tensorboard_logs")
    print(f"📊 Then open: http://localhost:6006")
    print(f"\n💡 Each training will appear with its model name in TensorBoard!")

if __name__ == "__main__":
    main()
