import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class SMSCategorizer:
    def __init__(self):
        self.categories = {
            'OTP': [],
            'Recovery': [],
            'Mambu': [],
            'Upsales': [],
            'Other': []
        }
        self.patterns = {
            'OTP': [
                r'fido security code',
                r'security code.*\d{4,8}',
                r'your.*code.*is.*\d{4,8}',
                r'^\d{4,8}$',
                r'security code with Fido is:\s*\d{6}.*Do not share this code with anyone.*For internal use: R/[A-Za-z0-9]+'
            ],
            'Recovery': [
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
                r'your payment is due on \d{2}-\d{2}-\d{4}',
                r'want an upgrade to ghc \d+',
                r'join fidobiz and submit your momo statement',
                r'hi \w+.*your payment is due',
                r'payment.*due.*upgrade.*ghc',
                r'fidobiz.*momo statement.*thank you',
                r'payment.*due.*\d{2}-\d{2}-\d{4}',
                r'upgrade.*ghc.*\d+',
                r'join.*fidobiz',
                r'submit.*momo statement',
                r'hi \w+.*payment.*due',
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
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.-]', ' ', text)
        return text.strip()
    
    def pattern_based_categorization(self, text):
        text_lower = text.lower()
        category_scores = {}
        category_priorities = {'Recovery': 3, 'Mambu': 2, 'Upsales': 1.5, 'OTP': 1}
        
        for category, patterns in self.patterns.items():
            score = 0
            for i, pattern in enumerate(patterns):
                if re.search(pattern, text_lower):
                    pattern_weight = len(patterns) - i
                    score += pattern_weight
            
            if score > 0:
                priority = category_priorities.get(category, 1)
                category_scores[category] = score * priority
        
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])[0]
            return best_category
        
        return 'Other'
    
    def extract_template(self, text):
        template = text
        fido_variables = [
            r'\{recipient_first_names\}', r'\{first_repayment_date\}', r'\{loan_amount\}',
            r'\{recipient_id\}', r'\{repayment_total_balance\}', r'\{payment_due_date\}',
            r'\{client first name\}', r'\{xxxxxxxx\}', r'\{transaction_amount\}',
            r'\{value_date\}', r'\{total_due\}', r'\{total_balance\}', r'\{first_name\}',
            r'\{discounted_amount\}', r'\{name\}', r'\{amount\}', r'\{number\}', r'\{date\}'
        ]
        
        for var in fido_variables:
            template = re.sub(var, '[VARIABLE]', template, flags=re.IGNORECASE)
        
        template = re.sub(r'\b\d{10,15}\b', '[PHONE]', template)
        template = re.sub(r'ghs?\s*\d+(?:\.\d{2})?', '[GHS_AMOUNT]', template, flags=re.IGNORECASE)
        template = re.sub(r'ghc?\s*\d+(?:\.\d{2})?', '[GHC_AMOUNT]', template, flags=re.IGNORECASE)
        template = re.sub(r'[\$£€¥₹]\s*\d+(?:\.\d{2})?', '[AMOUNT]', template)
        template = re.sub(r'\b\d+(?:\.\d{2})?\s*(?:dollars?|cents?|pounds?|euros?|naira|cedis?)\b', '[AMOUNT]', template)
        template = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', template)
        template = re.sub(r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b', '[DATE]', template, flags=re.IGNORECASE)
        template = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b', '[TIME]', template)
        template = re.sub(r'\b\d{1,2}am\s*-\s*\d{1,2}pm\b', '[TIME_RANGE]', template, flags=re.IGNORECASE)
        template = re.sub(r'client id:\s*\w+', 'client id: [CLIENT_ID]', template, flags=re.IGNORECASE)
        template = re.sub(r'\b\d+(?:\.\d+)?%', '[PERCENTAGE]', template)
        template = re.sub(r'\*\d+#', '[USSD_CODE]', template)
        template = re.sub(r'bit\.ly/\w+', '[BITLY_LINK]', template)
        template = re.sub(r'https?://[^\s]+', '[URL]', template)
        template = re.sub(r'\b\d+\s+(?:days?|weeks?|months?)\b', '[TIME_PERIOD]', template, flags=re.IGNORECASE)
        template = re.sub(r'\b\d{4,8}\b', '[OTP_CODE]', template)
        template = re.sub(r'_\s*_\s*_\s*_', '[PIN_PLACEHOLDER]', template)
        
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
        if not messages:
            return []
        
        processed_messages = [self.preprocess_text(msg) for msg in messages]
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(processed_messages)
            
            if n_clusters is None:
                n_clusters = min(10, max(2, len(messages) // 100))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            return clusters
        except:
            return [0] * len(messages)
    
    def analyze_sms_data(self, df, text_column='message', date_column=None):
        df['processed_message'] = df[text_column].apply(self.preprocess_text)
        df['category'] = df['processed_message'].apply(self.pattern_based_categorization)
        df['template'] = df['processed_message'].apply(self.extract_template)
        df['campaign_id'] = 0
        campaign_counter = 0
        
        for category in self.categories.keys():
            category_messages = df[df['category'] == category]['template'].tolist()
            if len(category_messages) > 1:
                clusters = self.cluster_similar_messages(category_messages)
                category_indices = df[df['category'] == category].index
                df.loc[category_indices, 'campaign_id'] = [c + campaign_counter for c in clusters]
                campaign_counter += max(clusters) + 1 if len(clusters) > 0 else 0
        
        return df
    
    def export_results(self, df, filename='sms_categorization_results.csv'):
        export_df = df[['processed_message', 'category', 'campaign_id', 'template']].copy()
        export_df.to_csv(filename, index=False)
