import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
import os

# Create directory if it doesn't exist
os.makedirs('assets', exist_ok=True)

# Sample data (replace with actual model results)
y_true = np.array([1] * 450 + [0] * 45 + [1] * 50 + [0] * 455)
y_pred = np.array([1] * 450 + [1] * 45 + [0] * 50 + [0] * 455)
y_scores = np.concatenate([
    np.random.beta(8, 2, 450),  # True positives
    np.random.beta(8, 2, 45),   # False positives
    np.random.beta(2, 8, 50),   # False negatives
    np.random.beta(2, 8, 455)   # True negatives
])

# 1. ROC Curve
plt.figure(figsize=(10, 6))
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = 0.96)')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('assets/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('assets/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Feature Importance
features = [
    'url_length', 'special_chars_count', 'domain_length',
    'num_dots', 'has_https', 'is_ip_address', 'domain_token_count',
    'digits_count', 'path_length', 'subdomain_length'
]
importances = [0.15, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]

plt.figure(figsize=(12, 6))
bars = plt.barh(features, importances)
plt.xlabel('Importance Score')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # Invert y-axis to show most important features at top

# Add value labels on the bars
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2,
             f'{width:.2f}', ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('assets/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Learning Curves
train_sizes = [1000, 2000, 3000, 4000, 5000]
train_scores = [0.98, 0.97, 0.96, 0.96, 0.96]
val_scores = [0.89, 0.91, 0.92, 0.93, 0.94]
test_scores = [0.88, 0.90, 0.91, 0.92, 0.92]

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, 'o-', label='Training Score', color='blue')
plt.plot(train_sizes, val_scores, 'o-', label='Validation Score', color='green')
plt.plot(train_sizes, test_scores, 'o-', label='Test Score', color='red')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('assets/learning_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations have been generated in the assets directory.")
