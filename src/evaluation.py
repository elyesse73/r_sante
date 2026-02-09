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
    visualize_explanation_simple
)

def evaluate_expert_comparison(model, loader, df_source, device):
    """
    Compares model predictions against Expert 1 and Expert 2 separately.

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
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Confusion Matrix: Expert 1
    cm1 = confusion_matrix(y_true_exp1, all_preds, labels=range(len(class_names)))
    # Normalize to percentages
    cm1_normalized = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm1_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0], vmin=0, vmax=1)
    axes[0].set_title("Confusion vs EXPERT 1 (Normalized)")
    axes[0].set_xlabel("Model Prediction")
    axes[0].set_ylabel("True Label (Expert 1)")

    # Confusion Matrix: Expert 2
    cm2 = confusion_matrix(y_true_exp2, all_preds, labels=range(len(class_names)))
    # Normalize to percentages
    cm2_normalized = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm2_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Confusion vs EXPERT 2 (Normalized)")
    axes[1].set_xlabel("Model Prediction")
    axes[1].set_ylabel("True Label (Expert 2)")

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
    - Group A: Experts agree (Consensus)
    - Group B: Experts disagree (Ambiguity)
    """
    model.eval()

    # Storage
    all_preds_idx = []
    all_confidences = [] # To see if the model "doubts"

    print("Group analysis in progress...")
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Get probabilities (Softmax) to check confidence
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            all_preds_idx.extend(preds.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    # Convert to numpy
    all_preds_idx = np.array(all_preds_idx)
    all_confidences = np.array(all_confidences)

    # --- 1. GROUP SEPARATION VIA DATAFRAME ---
    # Recreate masks based on source DataFrame (aligned with loader shuffle=False)
    mask_agreement = (df_source['expert1_label'] == df_source['expert2_label']).values
    mask_disagreement = ~mask_agreement

    print(f"\nDistribution: {mask_agreement.sum()} Agreement cases (A) / {mask_disagreement.sum()} Disagreement cases (B)")

    # --- 2. GROUP A ANALYSIS (AGREEMENT) ---
    # Here, the truth is simple: Expert 1 = Expert 2
    y_true_A = df_source.loc[mask_agreement, 'expert1_label'].map(loader.dataset.class_to_idx).values
    preds_A = all_preds_idx[mask_agreement]
    conf_A = all_confidences[mask_agreement]

    acc_A = accuracy_score(y_true_A, preds_A)
    print(f"\n✅ GROUP A (Consensus) - Accuracy : {acc_A*100:.2f}%")
    print(f"   -> Average Model Confidence : {conf_A.mean()*100:.1f}%")

    # --- 3. GROUP B ANALYSIS (DISAGREEMENT) ---
    # Here, we use "Relaxed Accuracy": Correct if model == Exp1 OR model == Exp2
    y_true_B1 = df_source.loc[mask_disagreement, 'expert1_label'].map(loader.dataset.class_to_idx).values
    y_true_B2 = df_source.loc[mask_disagreement, 'expert2_label'].map(loader.dataset.class_to_idx).values
    preds_B = all_preds_idx[mask_disagreement]
    conf_B = all_confidences[mask_disagreement]

    # Calculate Relaxed Accuracy: (Pred == Exp1) OR (Pred == Exp2)
    # Since Exp1 != Exp2, these conditions are mutually exclusive, we can sum booleans
    is_correct_relaxed = (preds_B == y_true_B1) | (preds_B == y_true_B2)
    acc_B_relaxed = is_correct_relaxed.mean()

    print(f"\n⚠️ GROUP B (Disagreement) - Relaxed Accuracy : {acc_B_relaxed*100:.2f}%")
    print(f"   (The model agrees with at least one of the two experts)")
    print(f"   -> Average Model Confidence : {conf_B.mean()*100:.1f}%")

    # --- 4. VISUALIZATION: LEARNED "DOUBT" ---
    plt.figure(figsize=(10, 6))
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

    # --- 2. GROUP ANALYSIS (Consensus vs Ambiguity) ---
    print("\n" + "="*40 + "\n2️⃣ GROUP PERFORMANCE (Agreement/Disagreement)\n" + "="*40)
    # Note: evaluate_by_groups must be defined in this file or imported
    evaluate_by_groups(model, test_loader, test_df, device)

    # --- 3. UNCERTAINTY (Entropy) ---
    print("\n" + "="*40 + "\n3️⃣ CALIBRATION & UNCERTAINTY (Entropy)\n" + "="*40)
    # Note: analyze_uncertainty must be defined in this file or imported
    analyze_uncertainty(model, test_loader, device)

    # --- 4. SOFT CONFUSION MATRIX (Relevant for Soft model, but works for both) ---
    print("\n" + "="*40 + "\n4️⃣ PROBABILISTIC CONFUSION MATRIX\n" + "="*40)
    # Note: plot_soft_confusion_matrix must be defined in this file or imported
    plot_soft_confusion_matrix(model, test_loader, device, dataset_classes)

    # --- 5. t-SNE VISUALIZATION (Latent Space) ---
    print("\n" + "="*40 + "\n5️⃣ t-SNE: SPACE REPRESENTATION\n" + "="*40)
    # We can limit num_samples if the test set is huge, otherwise None to take all
    # Note: Pass 'device' if your updated visualization function requires it
    visualize_tsne_dual_expert(model, test_loader, test_df, device, num_samples=None)

    print("\n" + "="*40 + "\n6️⃣ t-SNE: 'EYE' VISION (Features) vs 'BRAIN' (Classifier)\n" + "="*40)
    visualize_upstream_downstream_comparison(model, test_loader, test_df, device, num_samples=None)

    # --- 6. GRAD-CAM (Explainability on a few images) ---
    print("\n" + "="*40 + "\n7️⃣ GRAD-CAM: RANDOM EXAMPLES FROM TEST SET\n" + "="*40)
    
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