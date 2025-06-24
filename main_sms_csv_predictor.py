import pandas as pd
import re
from categorization import SMSCategorizer

def test_fido_patterns():
    """Test the categorization against actual Fido message templates"""
    
    # Actual Mambu templates from your data
    mambu_templates = [
        "Hi {RECIPIENT_FIRST_NAMES}, Your due date: {FIRST_REPAYMENT_DATE}. Check the Fido app Help section for payment steps.",
        "Hi {RECIPIENT_FIRST_NAMES} Your FIDO loan of{LOAN_AMOUNT} GHS (minus 1.6% commitment fee) is now in your mobile wallet. Client ID: {RECIPIENT_ID}.",
        "Hi {RECIPIENT_FIRST_NAMES}, Fido will NEVER ask you to repay a loan to a wallet number. Use only our in-app \"Make a Payment\" button or follow the repayment instructions in the app's \"Help\" section. Stay safe!",
        "Hi {RECIPIENT_FIRST_NAMES}, Your Fido loan is due! Pay GHS{REPAYMENT_TOTAL_BALANCE} by{PAYMENT_DUE_DATE} to stay eligible for future loans. Stay on track!",
        "Hello {Client First Name}, your Fido Loan is due tomorrow. This is a reminder for you to pay {TOTAL_BALANCE} to us any time tomorrow. Call us on 024 243 6885 if you need help on how to do this.",
        "Hi {RECIPIENT_FIRST_NAMES}! Payment of GHS{TRANSACTION_AMOUNT} confirmed{VALUE_DATE}. Next payment due soon - check your loan schedule.",
        "Hi {RECIPIENT_FIRST_NAMES}, Loan fully repaid! Want an upgrade to GHc 7500? Join FidoBiz and submit your momo statement!",
        "Hi {RECIPIENT_FIRST_NAMES}, Your account has been charged up to 22 GHS plus interest. Current balance:{TOTAL_DUE} GHS, with daily interest of 0.533%. Pay now to avoid extra fees!"
    ]
    
    # Actual Recovery templates from your data
    recovery_templates = [
        """Hello {first_name}, you have been offered a 50% DISCOUNT on your written off loan. Kindly pay Ghc{discounted_amount} in 4 weeks to enjoy this offer and have your name taken off the blacklist. T&C's applyTo repay your Fido Loan by MTN MoMo: first ensure you have enough money in your wallet.Dial: *170#Select: MoMoPay & Pay BillSelect: Pay BillSelect: General PaymentEnter Payment Code: FIDOEnter Amount:Enter Reference:Enter Pin Code: _ _ _ _Alert us if your payment has not reflected within 2 hours.A friend or MTN MoMo agent can pay on your behalf so long as they use your Client ID:""",
        "Hello {name} , Happy Holidays! Stay eligible for bigger loans by paying early and maintaining a strong Fido Score. Pay via 998# or the Fido app.",
        """Hello {name}, you have been offered a 50% DISCOUNT on your written off loan. Kindly pay {discounted_amount} in 4 weeks to enjoy this offer and have your name taken off the blacklist. T&amp;C's apply To repay your Fido Loan by TelecelCash, first ensure you have enough money in your wallet. Dial: *110# Select: Make Payments Select: Paybill Select: Other Payment options Enter Short Code: 633000 Enter Bill Number: Enter Amount: Enter Pin Code: _ _ _ _ You will receive a confirmation SMS from Telecel and FIDO. Alert us if your payment has not reflected within 2 hours.""",
        "Hello {name}, Early repayment reminder: Your loan is not due yet, but timely payment improves your Fido Score and keeps you eligible for your next loan. Pay via 998# or the Fido app",
        "Hello {name}, your loan is not due yet, but paying early boosts your FidoScore & keeps you eligible for future loans. Pay via 998# or the Fido app. Thank you!",
        "Hello, FidoBiz wants to serve you better! Tell us your preferred language for messages. It takes less than a minute. Click here https://forms.gle/CP9rHMLGzk9cGz437 https://tinyurl.com/fidobiz-survey",
        "Hello [name], FidoBiz would like to know how you prefer to receive our repayment reminders. Which language makes messages clearer for you—English or Twi? Click here to answer https://docs.google.com/forms/u/0/d/e/1FAIpQLScwNTelDKLybei_eS2gOhb63ffdfJgo4HTPrMmvC07H8RIehA/formResponse",
        "Hello {name}, your fidobiz loan amount of GHS {amount} is now {number} days overdue. We know things are tough but let's make a plan! You can pay in installments and enjoy a flexible plan. Start now! Don't delay!",
        "Settle things for the holiday! Clear your loan early to avoid last-minute delays or system issues. Settle before {date} for a smooth break. Pay via 998# or the Fido app. We are here to support you!",
        "Great news {name} ! Your exclusive 35% discount offer is still active. Pay only GHS {discounted_amount} by 6th June 2025 and the overdue balance will be cleared for you. Payment can be made in bits or in full by the deadline. For assistance, call 0540108694 from 7am - 4pm. T&C apply https://gh.fido.money/legal-pages/t-cs-legal-discounts",
        "Hi {name} , your payment is due soon. Paying early ensures a hassle-free process and can also increase your FidoScore. Pay Now!",
        "Help us serve you better! Tell us your preferred language for messages. It takes less than a minute. Click here: https://forms.gle/9JFQtK641CAFdiCu5",
        "Seize your 35% discount offer {name} . Clear the overdue balance by paying the discounted amount of GHS {discounted_amount} by 30th June 2025. After this, no more extensions. You can make payment on the Fido App, *998# or the mobile money USSD codes. For assistance, call 0540108694 from 7am - 4pm within working hours. T&C apply https://gh.fido.money/legal-pages/t-cs-legal-discounts. Thank you",
        "Dear {name} , your 35% discount offer ends today, 18th June 2025. Pay the discounted amount of GHS {discounted_amount} to have the overdue balance cleared. You can make payment on the Fido App, *998# or the mobile money USSD codes. For assistance, call 0540108694 from 7am - 4pm within working hours. T&C apply https://gh.fido.money/legal-pages/t-cs-legal-discounts.",
        "Dear {name} , your {number}% discount offer ends today, 17th June, 2025. Kindly pay the discounted amount of GHS {discounted_amount} to settle the overdue loan balance. T&Cs apply :https://gh.fido.money/legal-pages/t-cs-legal-discounts .Thank you"
    ]
    
    # Sample OTP messages
    otp_templates = [
        "Your Fido security code is 123456. Valid for 5 minutes.",
        "8998098"
    ]
    
    # Sample Upsales messages
    upsales_templates = [
        "Ready to grow? Upgrade to FidoBiz for larger business loans up to GHc 7500! Click here: bit.ly/47pns0u",
        "Earn GHC15! Refer friends and get paid: bit.ly/47pns0u",
        "Top up your account now! 50% bonus offer"
    ]
    
    # Initialize categorizer
    categorizer = SMSCategorizer()
    
    print("FIDO SMS PATTERN TESTING")
    print("=" * 50)
    
    # Test each category
    test_categories = [
        ("Mambu Templates", mambu_templates, "Mambu"),
        ("Recovery Templates", recovery_templates, "Recovery"),
        ("OTP Templates", otp_templates, "OTP"),
        ("Upsales Templates", upsales_templates, "Upsales")
    ]
    
    all_correct = 0
    all_total = 0
    
    for category_name, templates, expected_category in test_categories:
        print(f"\n{category_name}:")
        print("-" * 30)
        
        correct = 0
        total = len(templates)
        
        for template in templates:
            processed = categorizer.preprocess_text(template)
            predicted = categorizer.pattern_based_categorization(processed)
            extracted_template = categorizer.extract_template(processed)
            
            is_correct = predicted == expected_category
            if is_correct:
                correct += 1
            
            print(f"✓ {predicted}" if is_correct else f"✗ {predicted} (expected {expected_category})")
            print(f"  Original: {template[:80]}...")
            print(f"  Template: {extracted_template[:80]}...")
            print()
        
        accuracy = (correct / total) * 100
        print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        
        all_correct += correct
        all_total += total
    
    overall_accuracy = (all_correct / all_total) * 100
    print(f"\nOVERALL ACCURACY: {all_correct}/{all_total} ({overall_accuracy:.1f}%)")
    
    return all_correct, all_total

def analyze_pattern_coverage():
    """Analyze which patterns are most effective"""
    
    categorizer = SMSCategorizer()
    
    # Test messages with their expected categories
    test_cases = [
        ("Hi John, Your due date: 2024-12-15. Check the Fido app Help section for payment steps.", "Mambu"),
        ("Hello Peter, you have been offered a 50% DISCOUNT on your written off loan.", "Recovery"),
        ("Your Fido security code is 123456.", "OTP"),
        ("Upgrade to FidoBiz for larger business loans up to GHc 7500!", "Upsales"),
        ("Hi Sarah Your FIDO loan of 500 GHS (minus 1.6% commitment fee) is now in your mobile wallet.", "Mambu"),
        ("Hello James, your fidobiz loan amount of GHS 1200 is now 15 days overdue.", "Recovery"),
        ("Great news David! Your exclusive 35% discount offer is still active.", "Recovery"),
        ("Hi Anna! Payment of GHS200 confirmed 2024-12-01. Next payment due soon - check your loan schedule.", "Mambu")
    ]
    
    print("\nPATTERN COVERAGE ANALYSIS")
    print("=" * 50)
    
    for message, expected in test_cases:
        processed = categorizer.preprocess_text(message)
        predicted = categorizer.pattern_based_categorization(processed)
        
        print(f"\nMessage: {message[:60]}...")
        print(f"Expected: {expected}, Predicted: {predicted}")
        
        # Check which patterns matched for ALL categories
        all_matches = {}
        for category, patterns in categorizer.patterns.items():
            matched_patterns = []
            for pattern in patterns:
                if re.search(pattern, processed, re.IGNORECASE):
                    matched_patterns.append(pattern)
            if matched_patterns:
                all_matches[category] = matched_patterns
        
        if all_matches:
            print("All pattern matches:")
            for cat, patterns in all_matches.items():
                print(f"  {cat}: {patterns}")
        else:
            print("No patterns matched - needs new patterns!")
        
        # Show the processed text for debugging
        print(f"Processed text: {processed}")
        print("-" * 40)

def run_full_analysis_from_csv(csv_path: str, text_column: str, sample_size: int | None = None):
    """
    Loads SMS messages from a CSV, analyzes them, generates a report, and exports the results.
    """
    print(f"\n{'='*50}")
    print(f"RUNNING FULL ANALYSIS FROM CSV: {csv_path}")
    print("="*50)

    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df):,} messages.")
        if text_column not in df.columns:
            print(f"Error: Text column '{text_column}' not found in the CSV.")
            print(f"Available columns: {df.columns.tolist()}")
            return
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_path}'")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Optional: Use a smaller sample for faster testing
    if sample_size:
        print(f"Using a random sample of {sample_size:,} messages for this run.")
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    # Initialize and run the analysis
    categorizer = SMSCategorizer()
    analyzed_df = categorizer.analyze_sms_data(df, text_column=text_column)

    # Generate and print the report
    categorizer.generate_report(analyzed_df)

    # Export the detailed results
    output_filename = 'full_sms_categorization_results.csv'
    categorizer.export_results(analyzed_df, filename=output_filename)

if __name__ == "__main__":
    # --- You can uncomment the lines below to run the unit tests on the patterns ---
    #print("--- Running initial pattern tests ---")
    #test_fido_patterns()
    # analyze_pattern_coverage()

    # --- MAIN ANALYSIS ON YOUR CSV FILE ---
    # The code below is commented out to allow the pattern tests to run.
    # You can uncomment it again to run the full analysis on your CSV file.
    # Set your CSV file path and the column containing the SMS messages here
    FIDO_SMS_CSV_PATH = 'sms logs.csv'
    MESSAGE_COLUMN_NAME = 'text' 

    # To run on a smaller sample for speed (e.g., 5000 messages), 
    # add sample_size=5000 to the function call below.
    run_full_analysis_from_csv(
        csv_path=FIDO_SMS_CSV_PATH,
        text_column=MESSAGE_COLUMN_NAME
    )