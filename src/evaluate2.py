import json
import pandas as pd
import config
import ast

predictions_path = 'scripts/all-MimiLM-L6-v2/monolingual_predictions.json'
gs_path = 'scripts/all-MimiLM-L6-v2/processedData/'

# Load the JSON file with predictions
with open(predictions_path, 'r') as f:
    predictions = json.load(f)

# Load the CSV file with ground truth data
langs = config.LANGS

postsLang = {lang: 0 for lang in langs}
success_count_lang = {lang: 0 for lang in langs}

# Initialize variables to track success

for lang in langs:
    lang_gs = pd.read_csv(gs_path + f'posts_dev_{lang}.csv')

    lang_gs = lang_gs[['post_id', 'gs']]
    langpreds = predictions[lang].keys()

    # check that all the posts in the ground truth data are in the predictions and vice versa
    assert set(lang_gs['post_id'])  == set([int(langpred) for langpred in langpreds]), f"Missing posts in predictions for {lang}"
    
    postsLang[lang] = len(langpreds)

    for post_id, gs in lang_gs.values:
        gs = ast.literal_eval(gs)
        for idx in gs:
            if idx in predictions[lang][str(post_id)]:
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

