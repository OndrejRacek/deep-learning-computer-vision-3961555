import pandas as pd
import random

# Load original competition_train.csv
input_csv = "/home/vscode/.cache/kagglehub/datasets/vbookshelf/global-wheat-head-dataset-2021/versions/1/gwhd_2021/competition_train.csv"
df = pd.read_csv(input_csv)

# Group all boxes by image_name
grouped = df.groupby("image_name")["BoxesString"].apply(lambda x: ";".join(x)).reset_index()

# Generate random maturity labels per box
def generate_labels(boxes_str):
    box_count = boxes_str.count(";") + 1
    choices = ['young', 'mature', 'overripe']
    return ";".join(random.choice(choices) for _ in range(box_count))

grouped["maturity_label"] = grouped["BoxesString"].apply(generate_labels)

# Save the new CSV
output_path = input_csv.replace("competition_train.csv", "maturity_train.csv")
grouped.to_csv(output_path, index=False)

print(f"✅ maturity_train.csv created at:\n{output_path}")
