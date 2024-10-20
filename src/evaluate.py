import json
import pandas as pd

predictions_path = 'data/out/all-MiniLM-L6-v2/monolingual_predictions.json'
pairs_path = 'data/complete_data/pairs.csv'

# Load the JSON file with predictions
with open(predictions_path, 'r') as f:
    predictions = json.load(f)

# Load the CSV file with ground truth data
ground_truth = pd.read_csv(pairs_path)

# Initialize variables to track success
total_posts = 0
success_count = 0

# Loop through each post in the ground truth data
for _, row in ground_truth.iterrows():
    post_id = str(row['post_id'])
    ground_truth_fact_checked_id = row['fact_check_id']

    # Check if post ID exists in the predictions
    if post_id in predictions.keys():
        print('present')
        predicted_fact_checked_ids = predictions[post_id]

        # Check if the ground truth fact_checked_id is in the predicted list
        if ground_truth_fact_checked_id in predicted_fact_checked_ids:
            success_count += 1

    total_posts += 1

# Calculate the success rate
success_rate = success_count / total_posts if total_posts > 0 else 0

# Print results
print(f"Total posts evaluated: {total_posts}")
print(f"Successful matches: {success_count}")
print(f"Success rate: {success_rate:.2%}")
