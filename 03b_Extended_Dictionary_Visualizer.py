import pandas as pd
import matplotlib
import seaborn as sns

# Load all extended keywords with all cos thresholds
print("Loading extended keywords...")
ext_keywords_080 = pd.read_csv('data/dictionaries/extended_keywords_0.8.csv')
ext_keywords_090 = pd.read_csv('data/dictionaries/extended_keywords_0.9.csv')
ext_keywords_095 = pd.read_csv('data/dictionaries/extended_keywords_0.95.csv')
ext_keywords_097 = pd.read_csv('data/dictionaries/extended_keywords_0.97.csv')
ext_keywords_098 = pd.read_csv('data/dictionaries/extended_keywords_0.98.csv')
ext_keywords_099 = pd.read_csv('data/dictionaries/extended_keywords_0.99.csv')
print("Done.")

# Deduplicate
print("Deduplicating extended keywords...")
ext_keywords_080 = ext_keywords_080.drop_duplicates(subset=['keyword'])
ext_keywords_090 = ext_keywords_090.drop_duplicates(subset=['keyword'])
ext_keywords_095 = ext_keywords_095.drop_duplicates(subset=['keyword'])
ext_keywords_097 = ext_keywords_097.drop_duplicates(subset=['keyword'])
ext_keywords_098 = ext_keywords_098.drop_duplicates(subset=['keyword'])
ext_keywords_099 = ext_keywords_099.drop_duplicates(subset=['keyword'])
print("Done.")

# Make a dataframe with columns: cos_threshold, num_keywords, num_cso, num_method, num_task
num_keywords = []
num_cso = []
num_method = []
num_task = []
cos_threshold = ["0.80", "0.90", "0.95", "0.97", "0.98", "0.99"]

for ext_keywords in [ext_keywords_080, ext_keywords_090, ext_keywords_095, ext_keywords_097, ext_keywords_098, ext_keywords_099]:
    num_keywords.append(len(ext_keywords))
    num_cso.append(len(ext_keywords[ext_keywords['source'] == 'cso']))
    num_method.append(len(ext_keywords[ext_keywords['source'] == 'method']))
    num_task.append(len(ext_keywords[ext_keywords['source'] == 'task']))
    
print("Overview:")
df = pd.DataFrame({'cos_threshold': cos_threshold, 'num_keywords': num_keywords, 'num_cso': num_cso, 'num_method': num_method, 'num_task': num_task})

print(df)

# Plot the number of keywords per cos threshold
sns.set_style('whitegrid')
sns.set_context('paper')
sns.set_palette('colorblind')
matplotlib.rcParams['font.family'] = "serif"

ax = sns.lineplot(x='cos_threshold', y='num_keywords', data=df, marker='o')
ax.set_xlabel('Cosine similarity threshold')
ax.set_ylabel('Number of keywords')
# ax.set_title('Number of keywords per cosine similarity threshold')

# Set y-axis to logarithmic scale
ax.set_yscale('log')

# Save plot as svg
fig = ax.get_figure()
fig.savefig('plots/num_keywords_per_cos_threshold.svg', bbox_inches='tight')