import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import torch
from src.visualization import (
    visualize_tsne_dual_expert, 
    visualize_upstream_downstream_comparison, 
    visualize_explanation_simple,
    visualize_confidence_tsne
)

def evaluate_expert_comparison(model, loader, df_source, device):
    """
    Compares model predictions against Expert 1 and Expert 2 separately,
    and also compares Expert 1 against Expert 2.

    Args:
        model: Trained PyTorch model.
        loader: DataLoader (must be shuffle=False to match DataFrame order).
        df_source: The original DataFrame corresponding to the loader (val_df or test_df).
        device: 'cuda' or 'cpu'.
    """
    model.eval()
    all_preds = []

    print("Generating model predictions...")
    with torch.no_grad():
        for inputs, _ in loader: # We ignore the loader's soft labels here
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Get the model's hard decision (Index of the max probability)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)

    # --- Retrieve Ground Truths (from DataFrame) ---
    # We need the Name -> Index mapping to convert CSV string labels to integers
    class_to_idx = loader.dataset.class_to_idx
    class_names = loader.dataset.classes

    # Convert DataFrame string columns to numerical indices
    y_true_exp1 = df_source['expert1_label'].map(class_to_idx).values
    y_true_exp2 = df_source['expert2_label'].map(class_to_idx).values

    # --- REPORT EXPERT 1 ---
    print("\n" + "="*60)
    print("🔍 RESULTS vs EXPERT 1")
    print("="*60)
    print(classification_report(y_true_exp1, all_preds, target_names=class_names))

    # --- REPORT EXPERT 2 ---
    print("\n" + "="*60)
    print("🔍 RESULTS vs EXPERT 2")
    print("="*60)
    print(classification_report(y_true_exp2, all_preds, target_names=class_names))

    # --- COMPARATIVE VISUALIZATION ---
    # Changed to 3 columns to include Inter-Expert agreement
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # 1. Confusion Matrix: Model vs Expert 1
    cm1 = confusion_matrix(y_true_exp1, all_preds, labels=range(len(class_names)))
    # Normalize to percentages (adding epsilon to avoid division by zero)
    cm1_normalized = cm1.astype('float') / (cm1.sum(axis=1)[:, np.newaxis] + 1e-7)

    sns.heatmap(cm1_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=axes[0], vmin=0, vmax=1)
    axes[0].set_title("Model vs EXPERT 1 (Normalized)")
    axes[0].set_xlabel("Model Prediction")
    axes[0].set_ylabel("True Label (Expert 1)")

    # 2. Confusion Matrix: Model vs Expert 2
    cm2 = confusion_matrix(y_true_exp2, all_preds, labels=range(len(class_names)))
    # Normalize to percentages
    cm2_normalized = cm2.astype('float') / (cm2.sum(axis=1)[:, np.newaxis] + 1e-7)

    sns.heatmap(cm2_normalized, annot=True, fmt='.2%', cmap='Greens', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Model vs EXPERT 2 (Normalized)")
    axes[1].set_xlabel("Model Prediction")
    axes[1].set_ylabel("True Label (Expert 2)")

    # 3. Confusion Matrix: Expert 1 vs Expert 2 (Inter-Annotator Agreement)
    # We treat Expert 1 as "True" and Expert 2 as "Predicted" to see alignment
    cm3 = confusion_matrix(y_true_exp1, y_true_exp2, labels=range(len(class_names)))
    # Normalize to percentages
    cm3_normalized = cm3.astype('float') / (cm3.sum(axis=1)[:, np.newaxis] + 1e-7)

    sns.heatmap(cm3_normalized, annot=True, fmt='.2%', cmap='Oranges', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=axes[2], vmin=0, vmax=1)
    axes[2].set_title("INTER-EXPERT Agreement (Exp 1 vs Exp 2)")
    axes[2].set_xlabel("Expert 2 Label")
    axes[2].set_ylabel("Expert 1 Label")

    plt.tight_layout()
    plt.show()

def plot_soft_confusion_matrix(model, loader, device, classes):
    model.eval()

    # 1. Accumulate all probabilities and label vectors
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device) # Shape (Batch, Classes) e.g., [0, 0.5, 0.5, 0]

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu())

    # Concatenate everything -> (N_samples, N_classes)
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_labels, dim=0)

    # 2. The magic calculation (Matrix Product)
    # Transpose of True (Classes, N) x Pred (N, Classes) -> (Classes, Classes)
    cm_soft = torch.matmul(y_true.T, y_pred)

    # 3. Normalization (to get %)
    # Divide by the sum of real masses for each class (Total "Expert Mass")
    # .unsqueeze(1) allows dividing each row by its total
    cm_norm = cm_soft / y_true.sum(dim=0).unsqueeze(1)

    # Convert to Numpy for display
    cm_np = cm_norm.numpy()

    # 4. Heatmap Display
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_np, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title("Soft Confusion Matrix\n(Probability Mass Distribution)")
    plt.xlabel("Model Predicted Distribution")
    plt.ylabel("Truth Distribution (Experts)")
    plt.show()

def analyze_uncertainty(model, loader, device):
    model.eval()
    entropies_agreement = []
    entropies_disagreement = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            # Calculate entropy for each prediction (Model Uncertainty)
            # High entropy = flat distribution [0.3, 0.3, 0.4]
            # Low entropy = sharp peak [0.9, 0.05, 0.05]
            batch_entropies = entropy(probs, axis=1)

            # Separate based on whether experts agreed or not
            for i in range(len(labels)):
                max_val = labels[i].max()

                if max_val > 0.9: # Agreement (1.0)
                    entropies_agreement.append(batch_entropies[i])
                else: # Disagreement (0.5)
                    entropies_disagreement.append(batch_entropies[i])

    # Display
    plt.figure(figsize=(8, 6))
    plt.hist(entropies_agreement, bins=30, alpha=0.5, label='Experts Agreed (Consensus)', density=True)
    plt.hist(entropies_disagreement, bins=30, alpha=0.5, label='Experts Disagreed', density=True)
    plt.xlabel("Model Uncertainty (Entropy)")
    plt.ylabel("Density")
    plt.title("Does the model learn to doubt when necessary?")
    plt.legend()
    plt.show()

def evaluate_by_groups(model, loader, df_source, device):
    """
    Separate analysis:
    - Group A: Experts agree (Consensus) -> Measured by Top-1 Accuracy.
    - Group B: Experts disagree (Ambiguity) -> Measured by "Double Coverage" (do Top-2 match both experts?).
    """
    model.eval()

    # Storage
    all_top2_preds = [] # Store indices of top 2 predictions
    all_confidences = [] # Store probability of the #1 prediction (to measure doubt)

    print("Group analysis in progress...")
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Get Top-1 confidence (for the density plot)
            confidences, _ = torch.max(probs, dim=1)
            all_confidences.extend(confidences.cpu().numpy())

            # Get Top-2 indices (for the new metric)
            # top2_result shape: (Batch_Size, 2)
            _, top2_result = probs.topk(2, dim=1)
            all_top2_preds.extend(top2_result.cpu().numpy())

    # Convert to numpy
    all_top2_preds = np.array(all_top2_preds) # Shape (N, 2)
    all_confidences = np.array(all_confidences) # Shape (N,)

    # Extract Top-1 from Top-2 (it's just the first column) for Group A
    all_preds_top1 = all_top2_preds[:, 0]

    # --- 1. GROUP SEPARATION VIA DATAFRAME ---
    # Recreate masks based on source DataFrame (aligned with loader shuffle=False)
    mask_agreement = (df_source['expert1_label'] == df_source['expert2_label']).values
    mask_disagreement = ~mask_agreement

    print(f"\nDistribution: {mask_agreement.sum()} Agreement cases (A) / {mask_disagreement.sum()} Disagreement cases (B)")

    # --- 2. GROUP A ANALYSIS (AGREEMENT) ---
    # Metric: Standard Accuracy (Does Top-1 match the consensus?)
    y_true_A = df_source.loc[mask_agreement, 'expert1_label'].map(loader.dataset.class_to_idx).values
    preds_A = all_preds_top1[mask_agreement]
    conf_A = all_confidences[mask_agreement]

    acc_A = accuracy_score(y_true_A, preds_A)
    print(f"\n✅ GROUP A (Consensus) - Accuracy : {acc_A*100:.2f}%")
    print(f"   -> Average Model Confidence : {conf_A.mean()*100:.1f}%")

    # --- 3. GROUP B ANALYSIS (DISAGREEMENT) ---
    # Metric: Perfect Ambiguity Coverage (Do Top-2 predictions contain BOTH Expert 1 AND Expert 2?)
    
    # Get ground truths for Disagreement cases
    y_true_B1 = df_source.loc[mask_disagreement, 'expert1_label'].map(loader.dataset.class_to_idx).values
    y_true_B2 = df_source.loc[mask_disagreement, 'expert2_label'].map(loader.dataset.class_to_idx).values
    
    # Get Top-2 model predictions for these cases
    preds_top2_B = all_top2_preds[mask_disagreement] # Shape (N_B, 2)
    conf_B = all_confidences[mask_disagreement]

    # Logic: 
    # 1. Is Expert 1 in the model's Top 2?
    exp1_in_top2 = (preds_top2_B[:, 0] == y_true_B1) | (preds_top2_B[:, 1] == y_true_B1)
    
    # 2. Is Expert 2 in the model's Top 2?
    exp2_in_top2 = (preds_top2_B[:, 0] == y_true_B2) | (preds_top2_B[:, 1] == y_true_B2)
    
    # 3. STRICT CONDITION: BOTH must be true
    # This means the model predicted exactly the two classes chosen by the experts (in any order)
    perfect_coverage = exp1_in_top2 & exp2_in_top2
    
    acc_B_coverage = perfect_coverage.mean()

    print(f"\n⚠️ GROUP B (Disagreement) - Double Coverage Accuracy : {acc_B_coverage*100:.2f}%")
    print(f"   (The model's Top-2 predictions capture BOTH experts)")
    print(f"   -> Average Model Confidence : {conf_B.mean()*100:.1f}%")

    # --- 4. VISUALIZATION: LEARNED "DOUBT" ---
    plt.figure(figsize=(10, 6))
    # We clip at 0.999 to avoid graphic bugs with KDE if many values are exactly 1.0
    sns.kdeplot(conf_A, fill=True, label='Group A (Consensus)', color='green', clip=(0,1))
    sns.kdeplot(conf_B, fill=True, label='Group B (Disagreement)', color='orange', clip=(0,1))

    plt.title("Model Confidence Distribution\n(Did the model learn to doubt on difficult cases?)", fontsize=13)
    plt.xlabel("Prediction Confidence (Max Probability)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def run_full_test_battery(model, dataloaders, test_df, dataset_classes, device):
    """
    Executes the complete test suite on the provided model and test set.
    
    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataloaders (dict): Dictionary containing the 'test' DataLoader.
        test_df (pd.DataFrame): The DataFrame corresponding to the test set (for metadata).
        dataset_classes (list): List of class names (e.g., ['Anaphase', 'Metaphase'...]).
        device (torch.device): The device (CPU or CUDA) to run evaluation on.
    """
    # --- 0. PREPARATION ---
    print("📢 Launching full test battery on TEST SET...")
    
    # Ensure the best model is loaded
    # Note: This assumes 'best_model.pth' is in the current working directory
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        print("✅ 'best_model.pth' weights loaded successfully.")
    except FileNotFoundError:
        print("⚠️ Warning: 'best_model.pth' not found. Using current model weights.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load weights: {e}. Using current model weights.")

    model.eval()
    test_loader = dataloaders['test']

    # --- 1. METRICS & MATRICES (Classic vs Experts) ---
    print("\n" + "="*40 + "\n1️⃣ COMPARISON WITH EXPERTS (Matrices)\n" + "="*40)
    # Note: evaluate_expert_comparison must be defined in this file or imported
    evaluate_expert_comparison(model, test_loader, test_df, device)

    # --- 2. TOP-2 INDULGENT CONFUSION MATRICES ---
    print("\n" + "="*40 + "\n2️⃣ TOP-2 INDULGENT CONFUSION MATRICES\n" + "="*40)
    evaluate_top2_indulgent(model, test_loader, test_df, device)

    # --- 3. GROUP ANALYSIS (Consensus vs Ambiguity) ---
    print("\n" + "="*40 + "\n3️⃣ GROUP PERFORMANCE (Agreement/Disagreement)\n" + "="*40)
    # Note: evaluate_by_groups must be defined in this file or imported
    evaluate_by_groups(model, test_loader, test_df, device)

    # --- 4. SOFT CONFUSION MATRIX (Relevant for Soft model, but works for both) ---
    print("\n" + "="*40 + "\n4️⃣ PROBABILISTIC CONFUSION MATRIX\n" + "="*40)
    # Note: plot_soft_confusion_matrix must be defined in this file or imported
    plot_soft_confusion_matrix(model, test_loader, device, dataset_classes)

    # --- 5. t-SNE VISUALIZATION (Latent Space) ---
    print("\n" + "="*40 + "\n5️⃣ t-SNE: SPACE REPRESENTATION\n" + "="*40)
    # We can limit num_samples if the test set is huge, otherwise None to take all
    # Note: Pass 'device' if your updated visualization function requires it
    visualize_tsne_dual_expert(model, test_loader, test_df, num_samples=2000)

    # --- 6. t-SNE COMPARISON: FEATURES vs CLASSIFIER OUTPUTS ---
    print("\n" + "="*40 + "\n6️⃣ t-SNE: 'EYE' VISION (Features) vs 'BRAIN' (Classifier)\n" + "="*40)
    visualize_upstream_downstream_comparison(model, test_loader, test_df, num_samples=2000)

    # --- 7. t-SNE COLORED BY CONFIDENCE (Model's "Doubt") ---
    print("\n" + "="*40 + "\n7️⃣ t-SNE: CONFIDENCE LANDSCAPE\n" + "="*40)
    visualize_confidence_tsne(model, test_loader, test_df, device, num_samples=2000)

    # --- 8. GRAD-CAM (Explainability on a few images) ---
    print("\n" + "="*40 + "\n8️⃣ GRAD-CAM: RANDOM EXAMPLES FROM TEST SET\n" + "="*40)
    
    # We need access to the underlying dataset for visualization (to get raw images)
    test_dataset = test_loader.dataset
    
    # Displaying 3 random examples from the test set
    if len(test_dataset) > 0:
        indices = random.sample(range(len(test_dataset)), min(3, len(test_dataset)))
        for idx in indices:
            print(f"--- Explainability for Test image index {idx} ---")
            visualize_explanation_simple(model, test_dataset, idx, device)
    else:
        print("⚠️ Test dataset is empty, skipping Grad-CAM.")

def evaluate_top2_indulgent(model, loader, df_source, device):
    """
    Displays confusion matrices using "Top-2 Oracle" logic.
    
    Logic:
    - If the ground truth label is within the model's top 2 predictions, 
      we count it as a correct prediction (we 'force' the prediction to match the truth).
    - If the ground truth is NOT in the top 2, we keep the model's top-1 prediction 
      (the error is confirmed).

    This helps visualize if errors are "near misses" (hesitation between similar phases)
    or "complete misses".
    """
    model.eval()
    
    # Storage for the "corrected" predictions
    preds_indulgent_exp1 = [] 
    preds_indulgent_exp2 = [] 
    
    # Retrieve Ground Truths and Class Mappings
    class_to_idx = loader.dataset.class_to_idx
    class_names = loader.dataset.classes
    
    # Pre-load expert labels as integers to speed up the loop
    y_true_exp1 = df_source['expert1_label'].map(class_to_idx).values
    y_true_exp2 = df_source['expert2_label'].map(class_to_idx).values
    
    print("Generating Top-2 predictions...")
    
    counter = 0
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Get the top 2 class indices for each image
            # top2_preds.shape -> (Batch_Size, 2)
            _, top2_preds = outputs.topk(2, dim=1)
            top2_preds = top2_preds.cpu().numpy()
            
            # Iterate through the batch
            for i in range(len(top2_preds)):
                # Calculate the global index of the image in the dataset
                idx_global = counter + i
                
                # --- LOGIC VS EXPERT 1 ---
                true_label_1 = y_true_exp1[idx_global]
                
                # Check if the true label is in the top 2 predictions
                if true_label_1 in top2_preds[i]:
                    # Indulgent: We accept the prediction as correct
                    preds_indulgent_exp1.append(true_label_1)
                else:
                    # Strict: We keep the top-1 prediction (it's a miss)
                    preds_indulgent_exp1.append(top2_preds[i][0])
                    
                # --- LOGIC VS EXPERT 2 ---
                true_label_2 = y_true_exp2[idx_global]
                
                if true_label_2 in top2_preds[i]:
                    preds_indulgent_exp2.append(true_label_2)
                else:
                    preds_indulgent_exp2.append(top2_preds[i][0])
            
            counter += len(inputs)

    # --- VISUALIZATION ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Helper function to plot normalized confusion matrix
    def plot_cm(y_true, y_pred, ax, title, color_map):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        
        # Normalize to percentages (add epsilon to avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
        cm_norm = np.nan_to_num(cm_norm) # Replace NaNs with 0 if a class is empty
        
        # Calculate "Top-2 Accuracy" (Trace / Sum)
        acc = np.trace(cm) / np.sum(cm)
        
        sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap=color_map, cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax, vmin=0, vmax=1)
        ax.set_title(f"{title}\n(Top-2 Indulgent Acc: {acc*100:.2f}%)")
        ax.set_xlabel("Prediction (Top-2 Corrected)")
        ax.set_ylabel("True Label")

    # 1. Model vs Expert 1 (Indulgent)
    plot_cm(y_true_exp1, preds_indulgent_exp1, axes[0], 
            "Model vs EXPERT 1 (Top-2 Indulgent)", 'Purples')
    
    # 2. Model vs Expert 2 (Indulgent)
    plot_cm(y_true_exp2, preds_indulgent_exp2, axes[1], 
            "Model vs EXPERT 2 (Top-2 Indulgent)", 'Purples')

    plt.tight_layout()
    plt.show()