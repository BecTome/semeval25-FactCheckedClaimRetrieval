import json
import pandas as pd

predictions_path = 'scripts/all-MimiLM-L6-v2/monolingual_predictions.json'
pairs_path = 'data/complete_data/pairs.csv'

# Load the JSON file with predictions
with open(predictions_path, 'r') as f:
    predictions = json.load(f)

# Load the CSV file with ground truth data
ground_truth = pd.read_csv(pairs_path)
langs = ['fra', 'spa', 'eng', 'por', 'tha', 'deu', 'msa', 'ara']

postsLang = {lang: 0 for lang in langs}
success_count_lang = {lang: 0 for lang in langs}

# Initialize variables to track success

for lang in langs:
    langpreds = predictions[lang]
    postsLang[lang] = len(langpreds)
    # Loop through each post in the ground truth data
    for post_id in langpreds.keys():
        # Get the predicted fact-check for the current post
        pred_factchecks = langpreds[post_id]
        
        # Get the actual fact-check for the current post
        actual_factchecks = ground_truth[(ground_truth['post_id'] == int(post_id))]['fact_check_id']
        
        # Check if any of the actual fact-checks match any of the predicted fact-checks
        for actual_factcheck in actual_factchecks:
            if actual_factcheck in pred_factchecks:
                success_count_lang[lang] += 1
                
        
        

# Calculate the success rate
success_rate_lang = {lang: success_count_lang[lang] / postsLang[lang] if postsLang[lang] > 0 else 0 for lang in langs}

# Print results
print("Number of posts by language:")
for lang in langs:
    print(f"{lang}: {postsLang[lang]:.2f}")
print("Total number of posts:", sum(postsLang.values()), "\n")
print("Success rate by language:")
for lang in langs:
    print(f"{lang}: {success_count_lang[lang]} --> {success_rate_lang[lang]:.3f}%")
print("Total success rate:")
total_success_rate = sum(success_count_lang.values()) / sum(postsLang.values())
print(f"{total_success_rate:.3f}%")

