# 1. Sample Dataset
sample_dataset = """1.I absolutely love this phone!! battery lasts 2 days.
2.worst product ever :( money wasted !!
3.delivery was quick, but the packaging was damaged :( :(
4.excellent camera quality and amazing performance !!"""

print("Original Dataset:\n", sample_dataset)

# Step 1: Convert to lowercase
sample_dataset = sample_dataset.lower()
print("\nAfter Lowercasing:\n", sample_dataset)

# Step 2: Remove punctuation
import string
sample_dataset = sample_dataset.translate(str.maketrans('', '', string.punctuation))
print("\nAfter Punctuation Removal:\n", sample_dataset)

# Step 3: Remove numbers
import re
sample_dataset = re.sub(r'\d+', '', sample_dataset)
print("\nAfter Number Removal:\n", sample_dataset)

# Step 4: Remove stopwords
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
words = word_tokenize(sample_dataset)
filtered_words = [word for word in words if word.lower() not in stop_words]
sample_dataset = ' '.join(filtered_words)
print("\nAfter Stopword Removal:\n", sample_dataset)


# -------------------------------
# 2. Sample Email for Spam Detection
# -------------------------------

sample_email = """Subject: Urgent: Your account has been compromised!
Click this link now: http://malicious.com/phish123
Get FREE money! Limited time offer! Call 1-800-spam-now."""

print("\nOriginal Email:\n", sample_email)

# Step 1: Lowercase
sample_email = sample_email.lower()
print("\nAfter Lowercasing:\n", sample_email)

# Step 2: Remove punctuation
sample_email = sample_email.translate(str.maketrans('', '', string.punctuation))
print("\nAfter Punctuation Removal:\n", sample_email)

# Step 3: Remove numbers
sample_email = re.sub(r'\d+', '', sample_email)
print("\nAfter Number Removal:\n", sample_email)

# Step 4: Remove stopwords
email_words = word_tokenize(sample_email)
filtered_email_words = [word for word in email_words if word.lower() not in stop_words]
sample_email_processed = ' '.join(filtered_email_words)
print("\nAfter Stopword Removal:\n", sample_email_processed)
