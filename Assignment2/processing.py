import json
import pandas as pd

path = "basic_data_3.cleaned.jsonl"  # adjust to your filename
rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))
df = pd.DataFrame(rows)
print(df.head(20))
