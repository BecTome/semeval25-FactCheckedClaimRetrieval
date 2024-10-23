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

print(len([id for key in predictions.keys() for id in predictions[key]]))

postsLang = {lang: 0 for lang in langs}
success_count_lang = {lang: 0 for lang in langs}

# Initialize variables to track success

for lang in langs:
    langpreds = predictions[lang]
    postsLang[lang] = len(langpreds)
    # Loop through each post in the ground truth data
    for post_id in langpreds.keys():
        # Get the predicted fact-check for the current post
        pred_factcheck = langpreds[post_id]
        # print(pred_factcheck)
        # Get the actual fact-check for the current post
        actual_factchecks = ground_truth[(ground_truth['post_id'] == int(post_id))]['fact_check_id']
        # print(len(actual_factchecks))
        # Check if any of the actual fact-checks match any of the predicted fact-checks
        for actual_factcheck in actual_factchecks:
            if actual_factcheck in pred_factcheck:
                success_count_lang[lang] += 1
                
        
        

# Calculate the success rate
success_rate_lang = {lang: success_count_lang[lang] / postsLang[lang] if postsLang[lang] > 0 else 0 for lang in langs}

# Print results
print("Number of posts by language:")
for lang in langs:
    print(f"{lang}: {postsLang[lang]:.2f}")
print("Success rate by language:")
for lang in langs:
    print(f"{lang}: {success_rate_lang[lang]:.3f}%")

