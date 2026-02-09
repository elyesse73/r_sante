import os
import torch
import cv2
import torchstain
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from config import CONFIG

class MitosisDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        data : The filtered DataFrame containing only the relevant samples for this mode (AMF or NMF)
        root_dir : The root path for images (defined by config)
        mode : 'AMF' (Atypical) or 'NMF' (Normal)
        """

        self.data = data.reset_index(drop=True)
        self.transform = transform
        
        # --- AUTOMATIC CLASS DETECTION ---
        # We check which labels truly exist in this subset
        # e.g.: for AMF it will find 'anaphase lagging', 'bipolar asymmetry', etc.
        self.classes = sorted(self.data['expert1_label'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.labels1 = self.data['expert1_label'].map(lambda x: self.class_to_idx[x]).values
        self.labels2 = self.data['expert2_label'].map(lambda x: self.class_to_idx[x]).values
        self.num_classes = len(self.classes)

        print(f" Dataset initialized in [{CONFIG['MODE']}] mode")
        print(f"   -> {len(self.data)} images found.")
        print(f"   -> {len(self.classes)} classes : {self.class_to_idx}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Path construction (adapted to your file format)
        img_name = f"{row['dataset']}_{row['uid']}.png"
        img_path = os.path.join(self.root_dir, img_name)

        # Loading
        image = Image.open(img_path).convert('RGB')

        # Label
        # 3. Creation of Soft Label (This is where the magic happens)
        l1 = int(self.labels1[idx])
        l2 = int(self.labels2[idx])

        # Create a zero vector [0, 0, 0, 0, 0, 0, 0]
        label = torch.zeros(self.num_classes, dtype=torch.float32)

        # Add 0.5 for each expert's vote
        label[l1] += 0.5
        label[l2] += 0.5

        if self.transform:
            image = self.transform(image)

        return image, label
    
# --- 2. MACENKO AUGMENTATION ---
class MacenkoAugment(object):
    def __init__(self, ref_images_paths, p=0.5):
        """
        ref_images_paths : List of paths to various reference images.
        p : Probability of applying the transformation.
        """
        self.refs = []
        self.p = p

        for path in ref_images_paths:
            # Read OpenCV (BGR) -> Convert RGB
            if os.path.exists(path):
              img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
              T = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
              T.fit(img)
              self.refs.append(T)

    def __call__(self, img):
      # 1. Choose a random reference
      normalizer = np.random.choice(self.refs)
      # 2. Convert PIL -> Numpy
      img_np = np.array(img)
      # 3. Normalization
      norm, _, _ = normalizer.normalize(I=img_np, stains=True)
      # 4. Return PIL
      return Image.fromarray(norm.astype('uint8'))