import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.manifold import TSNE
from config import CONFIG

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