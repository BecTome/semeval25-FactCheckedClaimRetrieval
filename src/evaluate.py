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

# Initialize variables to track success
total_posts = 0
success_count = 0
for lang in langs:
    langpreds = predictions[lang]
    # Loop through each post in the ground truth data
    for post_id in langpreds.keys():
        total_posts += 1
        # Get the predicted fact-check for the current post
        pred_factcheck = langpreds[post_id]
        # Get the actual fact-check for the current post
        actual_factchecks = ground_truth[(ground_truth['post_id'] == post_id)]['fact_check_id'].values
        # Check if the predicted fact-check matches the actual fact-check
        if len(actual_factchecks) > 0 and pred_factcheck in actual_factchecks:
            success_count += 1

# Calculate the success rate
success_rate = success_count / total_posts if total_posts > 0 else 0

# Print results
print(f"Total posts evaluated: {total_posts}")
print(f"Successful matches: {success_count}")
print(f"Success rate: {success_rate:.2%}")
