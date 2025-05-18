import json

metadata_file = "beatport_metadata.json"
output_file = "tracklist.txt"

with open(metadata_file, "r") as f:
    metadata = json.load(f)

with open(output_file, "w", encoding="utf-8") as f:
    for track_title in metadata.keys():
        f.write(track_title + "\n")

print(f"Wrote {len(metadata)} track titles to {output_file}")