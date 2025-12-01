import json

with open("segmented_alex_pre_analysis_results.json", "r") as f:
    data = json.load(f)

for song in data["songs"]:
    seg_obj = song["segments"]
    new_segments = []

    for name, (start, end) in seg_obj.items():
        new_segments.append({
            "name": name,
            "start": start,
            "end": end
        })

    song["segments"] = new_segments

with open("segmented_alex_pre_analysis_results_converted.json", "w") as f:
    json.dump(data, f, indent=2)
