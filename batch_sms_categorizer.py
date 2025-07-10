import pandas as pd
import os
from pathlib import Path
from categorization import SMSCategorizer
import warnings
from datetime import datetime
import glob

def find_text_column(df):
    """Find the most likely text column in a DataFrame"""
    text_col_candidates = [col for col in df.columns if 'text' in col.lower() or 'message' in col.lower() or 'sms' in col.lower()]
    if text_col_candidates:
        return text_col_candidates[0]
    else:
        return df.columns[0]  # Return first column if no obvious text column found

def process_excel_file(file_path, categorizer):
    """Process a single Excel file and return categorized results"""
    try:
        # Read the Excel file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Find the text column
        text_column = find_text_column(df)
        
        # Remove rows with missing text
        df = df.dropna(subset=[str(text_column)])
        
        if len(df) == 0:
            print(f"⚠️  No valid data found in {file_path}")
            return None
        
        # Categorize messages
        print(f"📝 Processing {len(df)} messages from {os.path.basename(file_path)}...")
        
        categories = []
        for idx, message in enumerate(df[text_column]):
            category = categorizer.pattern_based_categorization(
                categorizer.preprocess_text(str(message))
            )
            categories.append(category)
            
            # Progress indicator for large files
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(df)} messages...")
        
        # Create results DataFrame
        results_df = df[[text_column]].copy()
        results_df['predicted_category'] = categories
        results_df['source_file'] = os.path.basename(file_path)
        results_df['processing_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"✅ Completed {os.path.basename(file_path)} - {len(results_df)} messages categorized")
        return results_df
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        return None

def batch_categorize_sms(folder_path, output_file=None):
    """Process all Excel/CSV files in a folder and combine results"""
    
    # Initialize categorizer
    print("🚀 Initializing SMS Categorizer...")
    categorizer = SMSCategorizer()
    
    # Find all Excel and CSV files in the folder
    folder = Path(folder_path)
    excel_files = []
    excel_files.extend(glob.glob(str(folder / "*.xlsx")))
    excel_files.extend(glob.glob(str(folder / "*.xls")))
    excel_files.extend(glob.glob(str(folder / "*.csv")))
    
    if not excel_files:
        print(f"❌ No Excel or CSV files found in {folder_path}")
        return None
    
    print(f"📁 Found {len(excel_files)} files to process:")
    for file in excel_files:
        print(f"   - {os.path.basename(file)}")
    
    # Process each file
    all_results = []
    successful_files = 0
    
    for i, file_path in enumerate(excel_files, 1):
        print(f"\n📊 Processing file {i}/{len(excel_files)}: {os.path.basename(file_path)}")
        
        results = process_excel_file(file_path, categorizer)
        if results is not None:
            all_results.append(results)
            successful_files += 1
    
    if not all_results:
        print("❌ No files were successfully processed")
        return None
    
    # Combine all results
    print(f"\n🔄 Combining results from {successful_files} files...")
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Add summary statistics
    print(f"📈 Total messages processed: {len(combined_df)}")
    print(f"📊 Categories found: {combined_df['predicted_category'].nunique()}")
    
    # Show category distribution
    category_counts = combined_df['predicted_category'].value_counts()
    print("\n📊 Category Distribution:")
    for category, count in category_counts.items():
        percentage = (count / len(combined_df)) * 100
        print(f"   {category}: {count} ({percentage:.1f}%)")
    
    # Save results
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"batch_categorized_sms_{timestamp}.csv"
    elif not output_file.endswith('.csv'):
        output_file += '.csv'
    
    combined_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    print(f"📋 To open: Right-click the file → 'Open with' → Excel or Google Sheets")
    
    return combined_df
def error_column_percentage(df, error_column="ErrorName"):
    """Calculate the percentage of messages with a non-empty ErrorName."""
    if error_column not in df.columns:
        return 0, 0, len(df)
    error_count = df[error_column].notna().sum()
    total = len(df)
    percentage = (error_count / total) * 100 if total > 0 else 0
    return percentage, error_count, total
def main():
    """Main function to run the batch categorizer"""
    print("=" * 60)
    print("📱 BATCH SMS CATEGORIZER")
    print("=" * 60)
    
    # Get folder path from user
    folder_path = "SMS_categorizor/tyr"
    
    if not os.path.exists(folder_path):
        print("❌ Folder does not exist!")
        return
    
    # Optional output file name
    output_file = input("Enter output file name (or press Enter for default): ").strip()
    if not output_file:
        output_file = None
    
    # Process files
    results = batch_categorize_sms(folder_path, output_file)
    
    if results is not None:
        print("\n🎉 Batch processing completed successfully!")
        print(f"📊 Final dataset contains {len(results)} messages from {results['source_file'].nunique()} files")
    else:
        print("\n❌ Batch processing failed!")

if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
    
    main() 