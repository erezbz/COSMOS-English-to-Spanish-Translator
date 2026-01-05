import csv

input_file = "sentencePairs.tsv"
output_file = "englishSpanish_Dataset.csv"

rows = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Expected format:
        # en_id \t english \t es_id \t spanish
        parts = line.split("\t")
        if len(parts) != 4:
            continue

        english = parts[1].strip()
        spanish = parts[3].strip()

        if english and spanish:
            rows.append((english, spanish))

# Write CSV
with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["english", "spanish"])
    writer.writerows(rows)

print(f"Saved {len(rows)} sentence pairs to {output_file}")
