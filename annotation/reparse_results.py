"""Re-parse existing results to fix JSON parsing without re-running inference."""

import json
import re
import sys
from pathlib import Path


def fix_trailing_commas(json_str):
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    return json_str


def parse_json_output(text):
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(fix_trailing_commas(match.group(1)))
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(fix_trailing_commas(match.group(0)))
        except json.JSONDecodeError:
            pass
    return None


results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/results")

# Look for per-image results in each_image_result/ subfolder or root
each_image_dir = results_dir / "each_image_result"
if each_image_dir.exists():
    search_dir = each_image_dir
else:
    search_dir = results_dir

files = sorted(search_dir.glob("*.json"))
fixed = 0
still_broken = 0

all_results = {}
for f in files:
    if f.name in ("summary.json", "_full_results.json", "eval_results.json"):
        continue
    d = json.load(open(f))
    if d.get("parsed_json") is None and d.get("stage2_raw_output"):
        parsed = parse_json_output(d["stage2_raw_output"])
        if parsed:
            d["parsed_json"] = parsed
            json.dump(d, open(f, "w"), indent=2, ensure_ascii=False)
            fixed += 1
        else:
            still_broken += 1
    key = d.get("annotation_key", f.stem)
    all_results[key] = d

# Update full results file
full_results_file = each_image_dir / "_full_results.json" if each_image_dir.exists() else results_dir / "summary.json"
if each_image_dir.exists():
    with open(full_results_file, "w") as fw:
        json.dump({"results": all_results}, fw, indent=2, ensure_ascii=False)
elif (results_dir / "summary.json").exists():
    summary = json.load(open(results_dir / "summary.json"))
    summary["results"] = all_results
    json.dump(summary, open(results_dir / "summary.json", "w"), indent=2, ensure_ascii=False)

print(f"Fixed: {fixed}, Still broken: {still_broken}, Total: {len(all_results)}")
