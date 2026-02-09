from datasets import load_dataset
import pandas as pd
import re

# Load WikiText-2 test set
dataset = load_dataset('wikitext', 'wikitext-2-v1', split='test')

# Extract sentences from the dataset
sentences = []
for text in dataset['text']:
    # Skip empty lines and section headers (lines starting with '=')
    if text.strip() and not text.strip().startswith('='):
        # Split by periods, question marks, and exclamation marks
        # This is a simple sentence splitter
        text_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        for sentence in text_sentences:
            if sentence.strip():  # Make sure sentence is not empty
                sentences.append(sentence.strip())
            
            if len(sentences) >= 200:
                break
    
    if len(sentences) >= 200:
        break

# Take exactly 200 sentences
sentences = sentences[:200]

# Create DataFrame and save to CSV
df = pd.DataFrame({'sentence': sentences})
df.to_csv('wikitext2_sentences.csv', index=False)

print(f"Saved {len(sentences)} sentences to wikitext2_sentences.csv")
print(f"\nFirst few sentences:")
for i, sent in enumerate(sentences[:3]):
    print(f"{i+1}. {sent}")