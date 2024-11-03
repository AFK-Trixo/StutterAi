import pandas as pd

# Load the labels file
labels_df = pd.read_csv('SEP-28k_labels.csv')

# List of stuttering type columns to convert to binary
stuttering_columns = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords', 'NaturalPause']

# Convert non-zero values to 1
for column in stuttering_columns:
    labels_df[column] = labels_df[column].apply(lambda x: 1 if x > 0 else 0)

# Save the updated labels to a new file
labels_df.to_csv('SEP-28k_labels_binary.csv', index=False)
