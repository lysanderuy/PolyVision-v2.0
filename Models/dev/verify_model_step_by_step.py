"""
Step-by-Step Model Verification and Copy Helper
Run this after copying .pth file from Docker to verify it works
"""

import torch
import os
import shutil
from datetime import datetime

def verify_pth_file(file_path):
    """Verify that a .pth file is valid and loadable"""
    
    print(f"🔍 STEP 6: Verifying {file_path}")
    print("-" * 50)
    
    # Check 1: File exists
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found at {file_path}")
        return False
    
    print(f"✅ File exists: {file_path}")
    
    # Check 2: File size
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"📊 File size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    
    if size_mb < 10:
        print("⚠️  WARNING: File seems small for a trained model (should be 100+ MB)")
    elif size_mb > 50:
        print("✅ Good size for a trained model")
    
    # Check 3: Try loading the file
    try:
        print("🔄 Loading model file...")
        
        # Load with CPU mapping (safer for testing)
        checkpoint = torch.load(file_path, map_location='cpu')
        print("✅ Model loaded successfully!")
        
        # Check structure
        if isinstance(checkpoint, dict):
            print(f"📋 Model structure: Dictionary with {len(checkpoint)} keys")
            
            # Show keys
            keys = list(checkpoint.keys())
            print(f"🔑 Keys: {keys[:5]}{'...' if len(keys) > 5 else ''}")
            
            # Check for model weights
            if 'model' in checkpoint:
                print("✅ Contains 'model' weights")
            elif any('conv' in str(k) or 'fc' in str(k) or 'layer' in str(k) for k in keys):
                print("✅ Contains neural network layers")
            else:
                print("⚠️  Unknown model structure")
        
        else:
            print("📋 Direct model object (not dictionary)")
        
        print("✅ MODEL FILE IS VALID!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        print("💡 The file may be corrupted during transfer")
        return False

def organize_model_file(source_path, model_type="binary"):
    """Organize model into proper directory structure"""
    
    if not os.path.exists(source_path):
        print(f"❌ Source file not found: {source_path}")
        return None
    
    print(f"\n📁 STEP 7: Organizing model file")
    print("-" * 50)
    
    # Create organized directory
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    if model_type == "binary":
        target_dir = f"SEAMaP-Binary-Full/faster_rcnn_R_50_FPN_3x/{timestamp}"
    else:
        target_dir = f"SEAMaP-Multi-class-100/faster_rcnn_R_50_FPN_3x/{timestamp}"
    
    # Create directory
    os.makedirs(target_dir, exist_ok=True)
    print(f"✅ Created directory: {target_dir}")
    
    # Copy model file
    target_path = os.path.join(target_dir, "model_final.pth")
    shutil.copy2(source_path, target_path)
    print(f"✅ Copied model to: {target_path}")
    
    # Create a simple info file
    info_file = os.path.join(target_dir, "model_info.txt")
    with open(info_file, "w") as f:
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Imported from Docker: {datetime.now()}\n")
        f.write(f"Source: {source_path}\n")
        f.write(f"Target: {target_path}\n")
    
    print(f"✅ Created info file: {info_file}")
    
    return target_path

def main():
    """Main verification process"""
    
    print("=" * 60)
    print("🔍 MODEL FILE VERIFICATION & ORGANIZATION")
    print("=" * 60)
    
    # Ask user for file path
    default_path = r"C:\Users\John Benedict\Desktop\PolyVision-2.0\Model-Version2\SEAMaP-Multi-class-100\faster_rcnn_R_50_FPN_3x\2025-10-01-03-54-34\model_final.pth"
    
    file_path = input(f"Enter .pth file path (or press Enter for default):\n[{default_path}]: ").strip()
    
    if not file_path:
        file_path = default_path
    
    # Verify the file
    if verify_pth_file(file_path):
        print(f"\n🎉 SUCCESS: Model file is valid!")
        
        # Ask if user wants to organize it
        organize = input("\nDo you want to organize this model into proper directory structure? (y/n): ").lower().strip()
        
        if organize.startswith('y'):
            model_type = input("Is this binary or multiclass model? (binary/multiclass): ").lower().strip()
            if model_type not in ['binary', 'multiclass']:
                model_type = 'binary'  # default
            
            organized_path = organize_model_file(file_path, model_type)
            
            if organized_path:
                print(f"\n✅ Model successfully organized at: {organized_path}")
                print("🎯 You can now use this model with your PolyVision application!")
        
    else:
        print(f"\n❌ FAILED: Model file has issues")
        print("💡 Try copying again from Docker or check the original file")

if __name__ == "__main__":
    main()