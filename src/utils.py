import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import v2, InterpolationMode
from src.config import CONFIG
from src.dataset import MacenkoAugment

# --- 4. TRANSFORMATION FACTORY (v2 Modernized & Merged Version) ---
def get_transforms(config, ref_paths=None):
    """
    Generates train/val transformations using the modern torchvision v2 API.
    Integrated with Macenko Augmentation logic.
    """

    # --- TRAINING PIPELINE ---
    train_ops = []

    # 1. Macenko Augmentation (Custom - works on PIL/Numpy)
    # We insert it BEFORE v2.ToImage() because Macenko class usually expects PIL/Numpy
    if CONFIG['USE_MACENKO'] and ref_paths is not None:
        train_ops.append(MacenkoAugment(ref_paths, p=0.5))

    # 2. New Transformations from partner
    train_ops.extend([
        v2.ToImage(),

        # Geometry: Flip and Rotation
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        # Rotation can create empty zones, fill=255 (white) for histology background
        v2.RandomRotation(degrees=180, fill=255),

        # Smart resizing to 224x224
        # scale=(0.8, 1.0): slight zoom
        # ratio=(0.95, 1.05): almost strict ratio conservation
        v2.RandomResizedCrop(
            size=(224, 224),
            scale=(0.8, 1.0),
            ratio=(0.95, 1.05),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        ),

        # Morphological deformation (tissue elasticity)
        v2.ElasticTransform(alpha=15.0, sigma=3.0, fill=255),

        # Color adjustments (Commented out as requested)
        #v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),

        # Conversion and noise
        v2.ToDtype(torch.float32, scale=True),
        v2.GaussianNoise(mean=0.0, sigma=0.01),

        # Final normalization
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- VALIDATION PIPELINE ---
    val_ops = [
        v2.ToImage(),
        # For validation, simply resize cleanly to 224
        v2.Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    return {
        'train': v2.Compose(train_ops),
        'val': v2.Compose(val_ops)
    }

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

def prepare_data_loaders(full_df, config, root_dir, get_transforms_fn, dataset_cls):
    """
    Orchestrates the entire data pipeline: filtering, splitting, transformations, 
    dataset creation, and dataloader generation.

    Args:
        full_df (pd.DataFrame): The raw dataframe loaded from CSV.
        config (dict): Configuration dictionary containing 'MODE', 'BATCH_SIZE', etc.
        root_dir (str): Path to the images directory.
        get_transforms_fn (callable): Function to get transforms (from src.transforms).
        dataset_cls (class): The Dataset class (from src.dataset).

    Returns:
        dict: A dictionary containing 'train', 'val', 'test' DataLoaders.
        tuple: (train_df, val_df, test_df) - The dataframes for each split (useful for analysis).
    """
    print(f"--- 🚀 STARTING DATA PREPARATION ({config['MODE']}) ---")

    # --- 1. INITIAL FILTERING ---
    # Determine if we are looking for Atypical or Normal based on MODE
    is_atypical = (config['MODE'] == 'AMF')
    
    # Filter by majority vote on atypia
    df_filtered = full_df[full_df['majority_atypical'] == is_atypical].copy()
    print(f"Total images for mode {config['MODE']} : {len(df_filtered)}")

    # Remove rare classes (less than 10 samples)
    counts = df_filtered['expert1_label'].value_counts()
    valid_classes = counts[counts >= 10].index
    df_final = df_filtered[df_filtered['expert1_label'].isin(valid_classes)].copy()

    dropped_classes = list(set(counts.index) - set(valid_classes))
    if dropped_classes:
        print(f"⚠️ Classes dropped because too rare : {dropped_classes}")

    # Keep only rows where labels strictly start with the prefix (e.g. 'NMF' or 'AMF')
    # This cleans up potential data entry errors in the CSV
    target_prefix = config['MODE']
    df_final = df_final[df_final['expert1_label'].str.startswith(target_prefix)]
    df_final = df_final[df_final['expert2_label'].str.startswith(target_prefix)]

    # --- 2. PREPARING STRATIFICATION ON SOFT VECTOR ---
    
    # Identify classes for the stratification logic
    all_classes = sorted(df_final['expert1_label'].unique())
    class_to_idx = {name: i for i, name in enumerate(all_classes)}
    num_classes = len(all_classes)

    # Helper function to create a "signature" string for stratification
    def get_soft_label_signature(row):
        vec = np.zeros(num_classes, dtype=np.float32)
        idx1 = class_to_idx.get(row['expert1_label'])
        idx2 = class_to_idx.get(row['expert2_label'])
        if idx1 is not None: vec[idx1] += 0.5
        if idx2 is not None: vec[idx2] += 0.5
        return "-".join(vec.astype(str))

    # Apply signature generation
    df_final['stratify_key'] = df_final.apply(get_soft_label_signature, axis=1)

    # Handling rare combinations for stratification safety
    # If a signature appears < 3 times, we cannot split it into Train/Val/Test
    key_counts = df_final['stratify_key'].value_counts()
    rare_keys = key_counts[key_counts < 3].index
    
    # Fallback: Stratify on expert1_label for these rare cases
    df_final.loc[df_final['stratify_key'].isin(rare_keys), 'stratify_key'] = \
        df_final.loc[df_final['stratify_key'].isin(rare_keys), 'expert1_label']

    print(f"Stratification key ready. Unique vectors: {len(key_counts)}")

    # Helper for Group Analysis (Agreement vs Disagreement)
    def assign_group(row):
        if row['expert1_label'] == row['expert2_label']:
            return 'Group A (Agreement)'
        else:
            return 'Group B (Disagreement)'

    df_final['expert_group'] = df_final.apply(assign_group, axis=1)

    # --- 3. SPLITTING (Train / Val / Test) ---
    # Split: 80% Train, 20% Temp
    train_df, temp_df = train_test_split(
        df_final,
        test_size=0.2,
        stratify=df_final['stratify_key'],
        random_state=42
    )

    # Split Temp: 50% Val, 50% Test (resulting in 10% each of total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['expert1_label'],
        random_state=42
    )

    print(f"Generated splits : Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # --- 4. PREPARE TRANSFORMS (Macenko) ---
    # Pick random reference images from the training set only
    random_refs = train_df.sample(n=5, random_state=42)
    ref_paths_list = [os.path.join(root_dir, f"{row['dataset']}_{row['uid']}.png") for _, row in random_refs.iterrows()]

    # Generate transforms dictionary
    data_transforms = get_transforms_fn(config, ref_paths=ref_paths_list)

    # --- 5. CREATE DATASETS ---
    # Note: Validate & Test use 'val' transforms (no augmentation)
    train_dataset = dataset_cls(train_df, root_dir=root_dir, transform=data_transforms['train'])
    val_dataset   = dataset_cls(val_df,   root_dir=root_dir, transform=data_transforms['val'])
    test_dataset  = dataset_cls(test_df,  root_dir=root_dir, transform=data_transforms['val'])

    # --- 6. SAMPLER (Class Imbalance Handling) ---
    sampler = None
    if config['USE_SAMPLER']:
        # Calculate weights based on Expert 1 labels in Train set
        class_counts = train_df['expert1_label'].value_counts().sort_index()
        weights = 1. / class_counts
        sample_weights = train_df['expert1_label'].map(weights).values
        
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(train_df),
            replacement=True
        )
        print("⚖️ WeightedRandomSampler activated.")

    # --- 7. DATALOADERS ---
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['BATCH_SIZE'], 
        sampler=sampler, 
        num_workers=config['NUM_WORKERS'],
        # Shuffle must be False if sampler is used (sampler does the shuffling)
        shuffle=(sampler is None) 
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['BATCH_SIZE'], 
        shuffle=False, 
        num_workers=config['NUM_WORKERS']
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['BATCH_SIZE'], 
        shuffle=False, 
        num_workers=config['NUM_WORKERS']
    )

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    print("✅ DataLoaders ready !")
    return dataloaders, (train_df, val_df, test_df), data_transforms

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, dataloaders, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(CONFIG['DEVICE'])
            labels = labels.to(CONFIG['DEVICE'])

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                train_dataset = dataloaders['train'].dataset
                ax.set_title(f'predicted: {train_dataset.classes[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)