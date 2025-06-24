import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class SMSCategorizer:
    def __init__(self):
        self.categories = {
            'OTP': [],
            'Recovery': [],
            'Mambu': [],
            'Upsales': [],
            'Other': []
        }
        
        # Define pattern rules for each category based on actual templates
        # Patterns are ordered by specificity (more specific patterns first)
        self.patterns = {
            'OTP': [
                r'fido security code',
                r'security code.*\d{4,8}',
                r'your.*code.*is.*\d{4,8}',
                r'^\d{4,8}$',  # Standalone code
            ],
            'Recovery': [
                 # High priority Recovery patterns
                 r'discount.*offer',
                 r'\d+%.*discount',
                 r'written off loan',
                 r'blacklist',
                 r'overdue.*balance',
                 r'discount.*amount',
                 r'fidobiz.*loan.*overdue',
                 r'fidobiz.*overdue',
                 r'\d+\s+days overdue',
                 r'exclusive.*discount',
                 r'discount offer.*active',
                 r'seize.*discount',
                 r'discount offer ends',
                 r'things are tough.*make a plan',
                 r'pay.*installments.*flexible',
                 # Lower priority
                 r'early repayment reminder',
                 r'paying early.*fidoscore',
                 r'loan.*not due yet',
                 r'preferred language',
                 r'serve you better',
                 r'happy holidays.*fido score',
                 r'clear.*loan early',
                 r'settle.*holiday'
            ],
            'Mambu': [
    # NEW HIGH PRIORITY PATTERNS (to catch misclassified messages)
    r'your payment is due on \d{2}-\d{2}-\d{4}',                    # Payment due with date format
    r'want an upgrade to ghc \d+',                                   # Upgrade offer pattern
    r'join fidobiz and submit your momo statement',                  # Fidobiz application process
    r'hi \w+.*your payment is due',                                  # Greeting + payment due
    r'payment.*due.*upgrade.*ghc',                                   # Payment due + upgrade offer
    r'fidobiz.*momo statement.*thank you',                          # Fidobiz workflow ending
    
    # ENHANCED EXISTING PATTERNS (made more flexible)
    r'payment.*due.*\d{2}-\d{2}-\d{4}',                            # More flexible payment due
    r'upgrade.*ghc.*\d+',                                            # More flexible upgrade pattern
    r'join.*fidobiz',                                                # Any fidobiz joining message
    r'submit.*momo statement',                                       # Momo statement submission
    r'hi \w+.*payment.*due',                                        # Personalized payment reminders
    
    # EXISTING HIGH PRIORITY PATTERNS (kept as-is)
    r'fido loan.*ghs.*commitment fee',
    r'loan.*ghs.*minus.*commitment fee',
    r'sent to your mobile wallet',
    r'mobile wallet.*client id',
    r'your due date.*fido app',
    r'fido loan is due.*pay ghs',
    r'payment.*ghs.*confirmed.*loan schedule',
    r'loan fully repaid',
    r'repay your loan easily.*momo',
    r'dial.*\*998#.*loan services',
    r'account.*charged.*ghs.*daily interest',
    r'lenders.*borrowers act.*court',
    r'installment loan.*due.*fido app',
    r'fido will never ask.*repay.*wallet',
    r'make a payment.*button.*help section',
    r'confirmation sms.*mtn.*fido',
    r'repayment.*due.*tomorrow.*penalty',
    r'settle.*debt.*lenders.*borrowers',
    r'next payment due.*loan schedule',
    r'current balance.*ghs.*interest',
    
    # EXISTING LOWER PRIORITY PATTERNS (kept as-is)
    r'your.*fido loan',
    r'client id.*\w+',
    r'fido app.*help',
    r'repayment.*ghs.*due',
    r'loan.*mobile wallet',
    r'stay eligible.*future loans',
    r'avoid penalty'
],
            'Upsales': [
                 r'top.?up'
            ]
        }
    
    def preprocess_text(self, text):
        """Clean and normalize text for analysis"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s.-]', ' ', text)
        
        return text.strip()
    
    def pattern_based_categorization(self, text):
        """Categorize based on predefined patterns with scoring"""
        text_lower = text.lower()
        
        # Score each category based on pattern matches
        category_scores = {}
        # Give higher priority to recovery messages as they are critical
        category_priorities = {'Recovery': 3, 'Mambu': 2, 'Upsales': 1.5, 'OTP': 1}
        
        for category, patterns in self.patterns.items():
            score = 0
            for i, pattern in enumerate(patterns):
                if re.search(pattern, text_lower):
                    # Higher score for patterns that appear earlier in the list (more specific)
                    pattern_weight = len(patterns) - i
                    score += pattern_weight
            
            if score > 0:
                # Apply category priority multiplier
                priority = category_priorities.get(category, 1)
                category_scores[category] = score * priority
        
        # Return category with highest score, or 'Other' if no matches
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])[0]
            return best_category
        
        return 'Other'
    
    def extract_template(self, text):
        """Extract template by replacing numbers and specific words with placeholders"""
        template = text
        
        # Replace Fido-specific template variables first
        fido_variables = [
            r'\{recipient_first_names\}', r'\{first_repayment_date\}', r'\{loan_amount\}',
            r'\{recipient_id\}', r'\{repayment_total_balance\}', r'\{payment_due_date\}',
            r'\{client first name\}', r'\{xxxxxxxx\}', r'\{transaction_amount\}',
            r'\{value_date\}', r'\{total_due\}', r'\{total_balance\}', r'\{first_name\}',
            r'\{discounted_amount\}', r'\{name\}', r'\{amount\}', r'\{number\}', r'\{date\}'
        ]
        
        for var in fido_variables:
            template = re.sub(var, '[VARIABLE]', template, flags=re.IGNORECASE)
        
        # Replace phone numbers
        template = re.sub(r'\b\d{10,15}\b', '[PHONE]', template)
        
        # Replace GHS amounts (Ghana Cedis)
        template = re.sub(r'ghs?\s*\d+(?:\.\d{2})?', '[GHS_AMOUNT]', template, flags=re.IGNORECASE)
        template = re.sub(r'ghc?\s*\d+(?:\.\d{2})?', '[GHC_AMOUNT]', template, flags=re.IGNORECASE)
        
        # Replace general amounts/currency
        template = re.sub(r'[\$£€¥₹]\s*\d+(?:\.\d{2})?', '[AMOUNT]', template)
        template = re.sub(r'\b\d+(?:\.\d{2})?\s*(?:dollars?|cents?|pounds?|euros?|naira|cedis?)\b', '[AMOUNT]', template)
        
        # Replace dates
        template = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', template)
        template = re.sub(r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b', '[DATE]', template, flags=re.IGNORECASE)
        
        # Replace times
        template = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b', '[TIME]', template)
        template = re.sub(r'\b\d{1,2}am\s*-\s*\d{1,2}pm\b', '[TIME_RANGE]', template, flags=re.IGNORECASE)
        
        # Replace Client IDs
        template = re.sub(r'client id:\s*\w+', 'client id: [CLIENT_ID]', template, flags=re.IGNORECASE)
        
        # Replace percentages
        template = re.sub(r'\b\d+(?:\.\d+)?%', '[PERCENTAGE]', template)
        
        # Replace MTN USSD codes and similar
        template = re.sub(r'\*\d+#', '[USSD_CODE]', template)
        
        # Replace URLs and bit.ly links
        template = re.sub(r'bit\.ly/\w+', '[BITLY_LINK]', template)
        template = re.sub(r'https?://[^\s]+', '[URL]', template)
        
        # Replace day counts (e.g., "5 days", "4 weeks")
        template = re.sub(r'\b\d+\s+(?:days?|weeks?|months?)\b', '[TIME_PERIOD]', template, flags=re.IGNORECASE)
        
        # Replace OTP codes (4-8 digits) - do this after other number replacements
        template = re.sub(r'\b\d{4,8}\b', '[OTP_CODE]', template)
        
        # Replace PIN patterns
        template = re.sub(r'_\s*_\s*_\s*_', '[PIN_PLACEHOLDER]', template)
        
        # Replace names (capitalized words that aren't common words)
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                       'your', 'you', 'dear', 'hi', 'hello', 'fido', 'mtn', 'momo', 'ghs', 'ghc',
                       'payment', 'loan', 'repay', 'app', 'sms', 'confirm', 'pay', 'amount'}
        words = template.split()
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if (clean_word not in common_words and len(clean_word) > 2 and 
                word and word[0].isupper() and not re.match(r'\[.*\]', word)):
                words[i] = '[NAME]'
        template = ' '.join(words)
        
        return template
    
    def cluster_similar_messages(self, messages, n_clusters=None):
        """Cluster messages based on text similarity"""
        if not messages:
            return []
        
        # Preprocess messages
        processed_messages = [self.preprocess_text(msg) for msg in messages]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(processed_messages)
            
            # Determine number of clusters if not specified
            if n_clusters is None:
                n_clusters = min(10, max(2, len(messages) // 100))
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            return clusters
        except:
            # If clustering fails, return all messages as one cluster
            return [0] * len(messages)
    
    def analyze_sms_data(self, df, text_column='message', date_column=None):
        """Main analysis function"""
        print(f"Analyzing {len(df)} SMS messages...")
        
        # Preprocess messages
        df['processed_message'] = df[text_column].apply(self.preprocess_text)
        
        # Pattern-based categorization
        print("Applying pattern-based categorization...")
        df['category'] = df['processed_message'].apply(self.pattern_based_categorization)
        
        # Extract templates for campaign identification
        print("Extracting message templates...")
        df['template'] = df['processed_message'].apply(self.extract_template)
        
        # Find similar campaigns within each category
        print("Clustering similar campaigns...")
        df['campaign_id'] = 0
        campaign_counter = 0
        
        for category in self.categories.keys():
            category_messages = df[df['category'] == category]['template'].tolist()
            if len(category_messages) > 1:
                clusters = self.cluster_similar_messages(category_messages)
                # Update campaign IDs
                category_indices = df[df['category'] == category].index
                df.loc[category_indices, 'campaign_id'] = [c + campaign_counter for c in clusters]
                campaign_counter += max(clusters) + 1 if len(clusters) > 0 else 0
        
        return df
    
    def generate_report(self, df):
        """Generate comprehensive analysis report"""
        print("\n" + "="*50)
        print("SMS CATEGORIZATION REPORT")
        print("="*50)
        
        # Overall statistics
        total_messages = len(df)
        print(f"\nTotal Messages Analyzed: {total_messages:,}")
        
        # Category breakdown
        print("\nCATEGORY BREAKDOWN:")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / total_messages) * 100
            print(f"  {category}: {count:,} ({percentage:.1f}%)")
        
        # Campaign analysis
        print("\nCAMPAIGN ANALYSIS:")
        unique_campaigns = df.groupby(['category', 'campaign_id']).size().reset_index(name='message_count')
        
        for category in self.categories.keys():
            if category in category_counts:
                cat_campaigns = unique_campaigns[unique_campaigns['category'] == category]
                print(f"\n{category} Campaigns:")
                print(f"  Total campaigns identified: {len(cat_campaigns)}")
                if len(cat_campaigns) > 0:
                    print(f"  Average messages per campaign: {cat_campaigns['message_count'].mean():.1f}")
                    print(f"  Largest campaign: {cat_campaigns['message_count'].max():,} messages")
        
        # Template examples
        print("\nSAMPLE TEMPLATES BY CATEGORY:")
        for category in self.categories.keys():
            if category in category_counts:
                templates = df[df['category'] == category]['template'].value_counts().head(3)
                print(f"\n{category}:")
                for template, count in templates.items():
                    print(f"  [{count:,} msgs] {template[:100]}...")
        
        return category_counts, unique_campaigns
    
    def export_results(self, df, filename='sms_categorization_results.csv'):
        """Export results to CSV"""
        export_df = df[['processed_message', 'category', 'campaign_id', 'template']].copy()
        export_df.to_csv(filename, index=False)
        print(f"\nResults exported to {filename}")

# Example usage and testing
if __name__ == "__main__":
    # Sample data for testing with actual Fido templates
    sample_messages = [
        "Hi John, Your due date: 2024-12-15. Check the Fido app Help section for payment steps.",
        "Hi Mary Your FIDO loan of 500 GHS (minus 1.6% commitment fee) is now in your mobile wallet. Client ID: FID123456.",
        "Your Fido security code is 123456. Valid for 5 minutes.",
        "Hello Peter, you have been offered a 50% DISCOUNT on your written off loan. Kindly pay Ghc250 in 4 weeks to enjoy this offer and have your name taken off the blacklist.",
        "Hi Sarah, Your Fido loan is due! Pay GHS350 by 2024-12-10 to stay eligible for future loans. Stay on track!",
        "Hello James, your fidobiz loan amount of GHS 1200 is now 15 days overdue. We know things are tough but let's make a plan!",
        "Hi Anna! Payment of GHS200 confirmed 2024-12-01. Next payment due soon - check your loan schedule. Ready to grow? Upgrade to FidoBiz for larger business loans up to GHc 7500!",
        "Top up your account now! 50% bonus offer",
        "Your verification code: 789012",
        "Hi Mike, Loan fully repaid! Want an upgrade to GHc 7500? Join FidoBiz and submit your momo statement!",
        "Hello Grace, Early repayment reminder: Your loan is not due yet, but timely payment improves your Fido Score and keeps you eligible for your next loan.",
        "Great news David! Your exclusive 35% discount offer is still active. Pay only GHS 180 by 6th June 2025 and the overdue balance will be cleared for you."
    ]
    
    # Create sample DataFrame
    sample_df = pd.DataFrame({
        'message': sample_messages,
        'timestamp': pd.date_range('2024-01-01', periods=len(sample_messages), freq='H')
    })
    
    # Initialize categorizer
    categorizer = SMSCategorizer()
    
    # Analyze sample data
    print("Testing SMS Categorizer with sample data...")
    analyzed_df = categorizer.analyze_sms_data(sample_df, text_column='message')
    
    # Generate report
    categorizer.generate_report(analyzed_df)
    
    print("\n" + "="*50)
    print("DETAILED RESULTS:")
    print("="*50)
    for idx, row in analyzed_df.iterrows():
        print(f"Message: {row['message'][:50]}...")
        print(f"Category: {row['category']}")
        print(f"Campaign ID: {row['campaign_id']}")
        print(f"Template: {row['template'][:50]}...")
        print("-" * 30)