import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.manifold import TSNE
from src.config import CONFIG

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False

def plot_array(fig, dataset, classes_to_plot=None, samples_per_class=7):
    # --- CORRECTION ---
    # Your dataset doesn't have .targets, we build it ourselves:
    # 1. We retrieve the textual labels from the internal DataFrame
    all_labels_str = dataset.data['expert1_label'].values

    # 2. We convert them to integers (indices) using the dataset's dictionary
    # This is necessary for the sorting logic below to work
    Y = np.array([dataset.class_to_idx[l] for l in all_labels_str])
    # ------------------

    if classes_to_plot is None:
        classes_to_plot = np.unique(Y)

    num_classes = len(classes_to_plot)

    for k, y in enumerate(classes_to_plot):
        # We look for all indices corresponding to class y
        idxs = np.flatnonzero(Y == y)

        # Safety: don't ask for more images than available
        num_samples_to_draw = min(len(idxs), samples_per_class)

        if num_samples_to_draw > 0:
            idxs = np.random.choice(idxs, num_samples_to_draw, replace=False)

            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + k + 1
                ax = fig.add_subplot(samples_per_class, num_classes, plt_idx)

                # Retrieve the image (Tuple: image, label)
                img_tensor, _ = dataset[idx]

                # Display management (Tensor C,H,W -> Numpy H,W,C)
                # We denormalize visually if necessary (here simple conversion)
                image = img_tensor.permute((1, 2, 0)).cpu().numpy()

                # If the image is normalized (negative or small values), rescale for display
                image = image - image.min()
                image = image / image.max()

                ax.imshow(image)
                ax.axis('off')

                # Column title for the first row
                if i == 0:
                    class_name = dataset.classes[y]
                    ax.set_title(class_name, fontsize=10)

def visualize_tsne_dual_expert(model, loader, df_source, num_samples=2000):
    """
    Generates a t-SNE and displays it twice:
    - Once colored according to Expert 1
    - Once colored according to Expert 2
    Marks disagreements with 'x' crosses.
    """
    model.eval()
    device = CONFIG['DEVICE']

    features_list = []

    # 1. Feature Extraction (Downstream / Vector)
    print("Extracting features...")
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)

            # Pass through the backbone
            feat_map = model.features(inputs)
            # Pooling + Classifier Start
            x = model.avgpool(feat_map)
            x = model.classifier[0](x) # LayerNorm
            feat_vec = model.classifier[1](x) # Flatten

            features_list.append(feat_vec.cpu())

            # Stop if enough points have been collected (to avoid t-SNE being too long)
            if num_samples and len(features_list) * inputs.size(0) >= num_samples:
                break

    # Concatenation
    X_features = torch.cat(features_list, dim=0).numpy()

    # Slice the DataFrame to match the number of extracted features exactly
    # (In case we stopped before the end of the loader)
    n_points = X_features.shape[0]
    df_subset = df_source.iloc[:n_points].copy()

    # 2. t-SNE Calculation (Only once!)
    print(f"Calculating t-SNE on {n_points} points...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_features)

    # 3. Plot Data Preparation
    # Identify disagreements to change the marker style
    disagreement_mask = df_subset['expert1_label'] != df_subset['expert2_label']
    df_subset['marker_style'] = np.where(disagreement_mask, 'Disagreement', 'Agreement')

    # Get class names for the legend
    class_order = sorted(df_source['expert1_label'].unique())

    # 4. Double Display
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    # --- LEFT PLOT: EXPERT 1 ---
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=df_subset['expert1_label'],
        style=df_subset['marker_style'], # Different shape for disagreement
        markers={'Agreement': 'o', 'Disagreement': 'X'},
        hue_order=class_order,
        palette='tab10', s=60, alpha=0.7, ax=axes[0]
    )
    axes[0].set_title("t-SNE Projection viewed by EXPERT 1\n('X' marks disagreements with Expert 2)", fontsize=13)
    axes[0].legend(loc='upper right', bbox_to_anchor=(1.25, 1), borderaxespad=0.)
    axes[0].grid(True, linestyle='--', alpha=0.3)

    # --- RIGHT PLOT: EXPERT 2 ---
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=df_subset['expert2_label'],
        style=df_subset['marker_style'],
        markers={'Agreement': 'o', 'Disagreement': 'X'},
        hue_order=class_order,
        palette='tab10', s=60, alpha=0.7, ax=axes[1]
    )
    axes[1].set_title("t-SNE Projection viewed by EXPERT 2\n(Geometry is identical, only colors change)", fontsize=13)
    # Hide the second legend to avoid visual redundancy, or leave it if desired
    axes[1].get_legend().remove()
    axes[1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

def visualize_upstream_downstream_comparison(model, loader, df_source, num_samples=2000):
    """
    Compares spatial organization (Upstream) vs Semantic (Downstream).
    Maintains marking of disagreements (Cross) to see where they are located.
    """
    model.eval()
    device = CONFIG['DEVICE']

    features_up_list = []
    features_down_list = []

    # 1. Extraction (Double extraction)
    print("Extracting Upstream AND Downstream features...")
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)

            # A. Upstream (Backbone Output)
            feat_map = model.features(inputs) # (Batch, 768, 7, 7)
            # Flatten for t-SNE: (Batch, 37632)
            feat_up_flat = feat_map.reshape(feat_map.size(0), -1)
            features_up_list.append(feat_up_flat.cpu())

            # B. Downstream (Just before classification)
            x = model.avgpool(feat_map)
            x = model.classifier[0](x) # LayerNorm
            feat_vec = model.classifier[1](x) # Flatten (Batch, 768)
            features_down_list.append(feat_vec.cpu())

            if num_samples and len(features_up_list) * inputs.size(0) >= num_samples:
                break

    # Convert to Numpy
    X_up = torch.cat(features_up_list, dim=0).numpy()
    X_down = torch.cat(features_down_list, dim=0).numpy()

    # Adjust DataFrame (in case we stopped before the end)
    n_points = X_up.shape[0]
    df_subset = df_source.iloc[:n_points].copy()

    # Prepare masks (Agreement/Disagreement)
    disagreement_mask = df_subset['expert1_label'] != df_subset['expert2_label']
    df_subset['marker_style'] = np.where(disagreement_mask, 'Disagreement', 'Agreement')
    class_order = sorted(df_source['expert1_label'].unique())

    # 2. Calculate both t-SNEs
    print("Computing Upstream t-SNE (The Eye)...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_emb_up = tsne.fit_transform(X_up)

    print("Computing Downstream t-SNE (The Brain)...")
    # Reinitialize t-SNE for the second run
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_emb_down = tsne.fit_transform(X_down)

    # 3. Comparative Display
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # Common style parameters
    scatter_kwargs = {
        'hue': df_subset['expert1_label'],
        'style': df_subset['marker_style'],
        'markers': {'Agreement': 'o', 'Disagreement': 'X'},
        'hue_order': class_order,
        'palette': 'tab10',
        's': 60,
        'alpha': 0.8
    }

    # --- PLOT 1: UPSTREAM ---
    sns.scatterplot(x=X_emb_up[:, 0], y=X_emb_up[:, 1], ax=axes[0], **scatter_kwargs)
    axes[0].set_title("1. Raw Vision (Upstream)\n(Organization based on visual similarity)", fontsize=14)
    axes[0].get_legend().remove() # Remove legend here to lighten the plot
    axes[0].grid(True, linestyle='--', alpha=0.3)

    # --- PLOT 2: DOWNSTREAM ---
    sns.scatterplot(x=X_emb_down[:, 0], y=X_emb_down[:, 1], ax=axes[1], **scatter_kwargs)
    axes[1].set_title("2. Semantic Vision (Downstream)\n(Organization based on biological class)", fontsize=14)
    # Keep legend only on the right
    axes[1].legend(loc='upper right', bbox_to_anchor=(1.2, 1), title="Classes (Expert 1)")
    axes[1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

def format_prediction_text(scores, class_names, top_k=3):
    """
    Formats prediction scores (or soft labels) into a readable string.

    Args:
        scores: A torch.Tensor (for soft labels) or numpy.ndarray (for probabilities).
        class_names: A list of class names.
        top_k: Number of top predictions to display for probabilities.
    """
    if isinstance(scores, torch.Tensor):
        # Handle soft labels (ground truth)
        active_classes_info = []
        for i, score in enumerate(scores):
            if score > 0: # Only consider classes that have a positive vote
                if scores.sum().item() == 1.0 and score.item() == 1.0: # If it's a hard label (sum is 1.0 and this is the only one)
                    active_classes_info.append(f"{class_names[i]}")
                elif scores.sum().item() > 0: # Soft labels with proportions
                    active_classes_info.append(f"{class_names[i]} ({score.item()*100:.0f}%)")
        return ", ".join(active_classes_info) if active_classes_info else "No Label"
    elif isinstance(scores, np.ndarray):
        # Handle predicted probabilities
        sorted_indices = np.argsort(scores)[::-1]
        top_predictions = []
        for i in range(min(top_k, len(sorted_indices))):
            idx = sorted_indices[i]
            prob = scores[idx]
            top_predictions.append(f"{class_names[idx]} ({prob*100:.2f}%)")
        return ", ".join(top_predictions)
    else:
        return "Invalid score format"

def visualize_explanation_simple(model, dataset, idx, device):
    """
    Cleaned up version: Simply displays the text above the image.
    """
    model.eval()

    if not HAS_GRAD_CAM:
        print(f"⚠️ [Info] Grad-CAM désactivé (librairie manquante). Affichage simple.")
        plt.figure(figsize=(6, 6))
        plt.imshow(img_display)
        plt.title(f"Image #{idx}\nVérité: {gt_text}\nPrédiction: {pred_text}")
        plt.axis('off')
        plt.show()
        return

    # --- CRITICAL FIX: Force gradient activation for Grad-CAM ---
    with torch.set_grad_enabled(True):

        # 1. Data
        img_tensor, label_vector = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # 2. Prediction
        output = model(input_tensor)
        # .detach() is important here because we activated gradients
        pred_probs = torch.softmax(output, dim=1).cpu().detach().numpy()[0]

        # 3. Texts
        # (Assuming format_prediction_text is defined in a previous cell)
        gt_text = format_prediction_text(label_vector, dataset.classes)
        pred_text = format_prediction_text(pred_probs, dataset.classes)

        # 4. Grad-CAM++ (With context manager to clean up hooks)
        target_layers = [model.features[-1][-1]]

        # Using 'with' to avoid conflicts if the cell is re-run
        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
            target_class = np.argmax(pred_probs)
            # Generate CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    # 5. Denormalization (Outside gradient calculation, for display only)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_display = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1)

    visualization = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)

    # --- DISPLAY (Identical to your request) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # LEFT: TRUTH
    axes[0].imshow(img_display)
    axes[0].set_title(f"Image #{idx} - Truth (Experts)\n\n{gt_text}",
                      fontsize=11, loc='left')
    axes[0].axis('off')

    # RIGHT: PREDICTION
    axes[1].imshow(visualization)
    axes[1].set_title(f"Grad-CAM++ (Focus: {dataset.classes[target_class]})\n\n{pred_text}",
                      fontsize=11, loc='left', color='darkblue')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_confidence_tsne(model, loader, df_source, device, num_samples=2000):
    """
    Generates a t-SNE visualization colored by the model's confidence.
    
    Visual Encoding:
    - Style: Matches the "Seaborn/Paper" aesthetic (White background, soft grid).
    - Color (Gradient): Red-Yellow-Green Colormap.
      (Red = Low Confidence/Doubt, Green = High Confidence/Certainty)
    - Shape:
      o (Circle) = Experts Agree (Consensus)
      X (Cross)  = Experts Disagree (Ambiguity)
      
    Hypothesis: 
    We expect high confidence (Green) in cluster centers and low confidence 
    (Red/Orange + Crosses) at the boundaries between clusters.
    """
    model.eval()
    
    features_list = []
    confidences_list = []
    
    print(f"Extracting features & confidence for t-SNE (Max samples: {num_samples})...")
    
    count = 0
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            
            # 1. Extract Features (for t-SNE coordinates)
            # Assuming ConvNeXt architecture
            feat_map = model.features(inputs)
            x = model.avgpool(feat_map)
            # Flatten for the linear layer
            x_flat = model.classifier[0](x) # LayerNorm/Flatten usually
            
            # Use penultimate layer features
            features_list.append(model.classifier[1](x_flat).cpu()) 
            
            # 2. Extract Confidence (for Color)
            # We run the full forward pass to get probabilities
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            # We take the probability of the predicted class (Max prob)
            max_probs, _ = torch.max(probs, dim=1)
            confidences_list.append(max_probs.cpu())
            
            count += inputs.size(0)
            if num_samples and count >= num_samples:
                break
                
    # Concatenate results
    X_features = torch.cat(features_list, dim=0).numpy()
    confidences = torch.cat(confidences_list, dim=0).numpy()
    
    # Align DataFrame with the subset processed
    df_subset = df_source.iloc[:X_features.shape[0]].copy()
    
    # Create Agreement Mask
    agreement_mask = (df_subset['expert1_label'] == df_subset['expert2_label']).values
    
    print("Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_features)
    
    # --- PLOTTING (Refined Aesthetic) ---
    
    # Force the style context to ensure the look matches "masterclass.png"
    # 'seaborn-whitegrid' gives the white background + grey grid lines
    try:
        style_context = 'seaborn-v0_8-whitegrid'
    except:
        style_context = 'seaborn-whitegrid' # Fallback for older matplotlib versions

    with plt.style.context(style_context): 
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 1. Define the Colormap (Red -> Yellow -> Green)
        # Red = Doubt (0.0), Green = Certainty (1.0)
        cmap_name = 'RdYlGn' 
        
        # 2. Plot Consensus Points (Circles) - Background Layer
        sc1 = ax.scatter(
            X_embedded[agreement_mask, 0], 
            X_embedded[agreement_mask, 1],
            c=confidences[agreement_mask],
            cmap=cmap_name,
            marker='o',
            s=70,          # Size
            alpha=0.8,     # Opacity
            label='Consensus (Agree)',
            edgecolors='white', # White edge makes it look clean (like stickers)
            linewidth=0.8,
            vmin=0.4, vmax=1.0  # Clip colors: <40% is pure red, >100% is pure green
        )
        
        # 3. Plot Disagreement Points (Crosses) - Foreground Layer
        sc2 = ax.scatter(
            X_embedded[~agreement_mask, 0], 
            X_embedded[~agreement_mask, 1],
            c=confidences[~agreement_mask],
            cmap=cmap_name,
            marker='X',
            s=90,          # Slightly larger to pop out
            alpha=1.0,     # Full opacity
            label='Ambiguity (Disagree)',
            edgecolors='k', # Thin black edge to define the cross shape clearly
            linewidth=0.5,
            vmin=0.4, vmax=1.0
        )
        
        # 4. Decoration matching the reference image
        ax.grid(True, linestyle='--', alpha=0.4) # Dashed, soft grid
        ax.set_title("t-SNE learned Space colored by Model Confidence\n(Red = Doubt, Green = Certainty)", fontsize=16, pad=20)
        
        # Colorbar configuration (Modern look)
        cbar = plt.colorbar(sc1, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Model Confidence (Probability)', rotation=270, labelpad=20, fontsize=12)
        cbar.outline.set_visible(False) 
        
        # Legend configuration
        legend = ax.legend(loc='upper right', frameon=True, fontsize=12, fancybox=True, framealpha=0.9)
        legend.get_frame().set_edgecolor('lightgray')

        plt.tight_layout()
        plt.show()