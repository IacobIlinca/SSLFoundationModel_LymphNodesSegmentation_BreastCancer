import json


with open("../masks/all_not_lymph_terms.json", "r", encoding="utf-8") as f:
    terms1 = json.load(f)["terms"]

with open("../masks/lymph_terms.json", "r", encoding="utf-8") as f:
    terms2 = json.load(f)["terms"]


matches = []

for t1 in terms1:
    for t2 in terms2:
        if t1 in t2 or t2 in t1:
            matches.append((t1, t2))


for a, b in matches:
    print(f"{a}  <->  {b}")

print(f"\nTotal matches: {len(matches)}")