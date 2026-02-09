import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def load_and_analyze_metadata(csv_path):
    """
    Loads the dataset CSV, computes agreement statistics between experts, 
    and prints a summary report of the dataset distribution.

    Args:
        csv_path (str): Path to the AMI-BR.csv file.

    Returns:
        pd.DataFrame: The dataframe enriched with agreement and atypical analysis columns.
    """
    # 1. Load and copy to avoid altering the original
    df_stats = pd.read_csv(csv_path).copy()

    # 2. Create agreement analysis columns
    # Phase agreement (expert 1 vs expert 2)
    df_stats['phase_agreement'] = df_stats['expert1_label'] == df_stats['expert2_label']

    # Calculate the number of experts who voted "Atypical = True"
    # Sum booleans (True=1, False=0)
    atypical_cols = ['expert1_atypical', 'expert2_atypical', 'expert3_atypical']
    df_stats['nb_atypical_votes'] = df_stats[atypical_cols].sum(axis=1)

    # Define confidence levels for atypia
    df_stats['atypical_consensus'] = df_stats['nb_atypical_votes'] == 3
    df_stats['atypical_majority'] = df_stats['nb_atypical_votes'] >= 2
    df_stats['atypical_disagreement'] = (df_stats['nb_atypical_votes'] > 0) & (df_stats['nb_atypical_votes'] < 3)

    # 3. Display Global Statistics
    total = len(df_stats)
    print(f"--- DATASET ANALYSIS ({total} cells) ---")

    # Phase agreement
    match_rate = df_stats['phase_agreement'].mean() * 100
    print(f"\n[PHASES] Expert 1 / Expert 2 Agreement : {match_rate:.2f}%")

    # Atypia analysis (Majority vs Consensus)
    print("\n[ATYPIA] Vote distribution :")
    atypical_stats = {
        "Consensus (3/3 experts)": df_stats['atypical_consensus'].sum(),
        "Simple majority (>= 2/3)": df_stats['majority_atypical'].sum(), # Uses original column
        "Total disagreement (1 or 2 votes)": df_stats['atypical_disagreement'].sum(),
        "Strictly Normal (0/3 votes)": (df_stats['nb_atypical_votes'] == 0).sum()
    }

    for label, count in atypical_stats.items():
        print(f"- {label}: {count} ({count/total*100:.1f}%)")

    # 4. Top 5 phase confusions (when experts disagree)
    print("\n[CONFUSIONS] Top 5 most frequent phase disagreements :")
    disagreements = df_stats[df_stats['phase_agreement'] == False]
    confusion = disagreements.groupby(['expert1_label', 'expert2_label']).size().sort_values(ascending=False)
    print(confusion.head(5))

    # 5. Class distribution (based on majority for atypia)
    print("\n[CLASSES] Final distribution (based on Expert 1) :")
    print(df_stats['expert1_label'].value_counts(normalize=True) * 100)

    return df_stats

def compare_experts_agreement(df):
    """
    Compares the labels provided by Expert 1 and Expert 2.
    It generates a classification report considering Expert 1 as the Ground Truth
    and Expert 2 as the prediction (to measure agreement), and displays a normalized confusion matrix.

    Args:
        df (pd.DataFrame): The dataframe containing 'expert1_label' and 'expert2_label' columns.
    """
    # 1. Extract labels from expert 1 and expert 2 from the DataFrame
    expert1_labels = df['expert1_label'].values
    expert2_labels = df['expert2_label'].values

    # 2. Use the class_name -> index mapping based on ALL unique classes in the dataset.
    all_class_names = sorted(df['expert1_label'].unique())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(all_class_names)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Convert string labels to numerical labels using the mapping
    numerical_expert1_labels = np.array([class_to_idx[label] for label in expert1_labels if label in class_to_idx])
    numerical_expert2_labels = np.array([class_to_idx[label] for label in expert2_labels if label in class_to_idx])

    # Ensure lengths are identical after filtering if necessary
    if len(numerical_expert1_labels) != len(numerical_expert2_labels):
        print("Warning: Numerical label lists for expert 1 and expert 2 have different lengths.")
        min_len = min(len(numerical_expert1_labels), len(numerical_expert2_labels))
        numerical_expert1_labels = numerical_expert1_labels[:min_len]
        numerical_expert2_labels = numerical_expert2_labels[:min_len]

    # Get the unique numerical classes present in both experts' labels
    unique_labels_present = sorted(list(np.unique(np.concatenate((numerical_expert1_labels, numerical_expert2_labels)))))

    # Map these numerical labels to their corresponding class names for display
    report_target_names = [idx_to_class[idx] for idx in unique_labels_present]

    print(f"Comparison of Expert 2 labels (true) vs Expert 1 (predicted for comparison) on {len(numerical_expert1_labels)} samples of the full dataset :")
    # Note: Using labels=unique_labels_present ensures we only report on classes that actually appear
    print(classification_report(numerical_expert1_labels, numerical_expert2_labels, target_names=report_target_names, labels=unique_labels_present))

    # Confusion Matrix
    # `y_true` is expert 1, `y_pred` is expert 2
    cm = confusion_matrix(numerical_expert1_labels, numerical_expert2_labels, labels=unique_labels_present)

    # Normalization to get percentages
    # Handle case where axis sum is zero to avoid division by zero
    cm_normalized = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        row_sum = cm[i, :].sum()
        if row_sum > 0:
            cm_normalized[i, :] = cm[i, :].astype('float') / row_sum

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=report_target_names, yticklabels=report_target_names, vmin=0, vmax=1)
    plt.title("Confusion Matrix: Expert 2 (Predicted) vs Expert 1 (True) - Full Dataset")
    plt.xlabel("Expert 2 Label")
    plt.ylabel("Expert 1 Label")
    plt.tight_layout()
    plt.show()