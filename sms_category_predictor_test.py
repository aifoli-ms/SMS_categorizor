import pandas as pd
from categorization import SMSCategorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Set your input CSV file name here
input_csv = 'sampled_sms_data.csv'  # Change this to your actual file
message_column = 'processed_message'  # Name of the column with the message text
category_column = 'Category'  # Name of the column with the true category

# Load the data
try:
    df = pd.read_csv(input_csv)
except Exception as e:
    raise FileNotFoundError(f"Could not load file: {e}")

# Check columns
if message_column not in df.columns or category_column not in df.columns:
    raise ValueError(f"Input CSV must contain columns '{message_column}' and '{category_column}'")

# Initialize categorizer
categorizer = SMSCategorizer()

# Predict categories
print("Predicting categories...")
df['predicted_category'] = df[message_column].apply(lambda x: categorizer.pattern_based_categorization(categorizer.preprocess_text(x)))

# Calculate accuracy
accuracy = accuracy_score(df[category_column], df['predicted_category'])
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Print classification report (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(df[category_column], df['predicted_category'])) 

# Create a DataFrame with message, true category, and predicted category
output_df = df[[message_column, category_column, 'predicted_category']].copy()
output_df.to_csv('predictions_output.csv', index=False)
print("\nSample of predictions:")
print(output_df.head())


# Find messages where true label is Mambu but predicted as Other
misclassified_mambu_as_other = df[
    (df['Category'] == 'Mambu') &
    (df['predicted_category'] == 'Other')
]

# Display the first few rows
print(misclassified_mambu_as_other.head())



# Define label order (use sorted list of unique labels or manually define)
labels = sorted(df[category_column].unique())

# Create confusion matrix
cm = confusion_matrix(df[category_column], df['predicted_category'], labels=labels)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', xticks_rotation="vertical")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()