import json, os
from pprint import pprint

input_path = "./output.jsonl"
output_path = "./output_cleaned.jsonl"

with open(input_path, "r", encoding="utf-8") as reader, \
     open(output_path, "a+", encoding="utf-8") as writer:
    for raw_line in reader:
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        record.pop('SUNGSANG', None)
        record.pop('STTEMNT_NO', None)
        record.pop('REGIST_DT', None)

        for key in list(record.keys()):
            if record[key] is None:
                record[key] = ""

        writer.write(json.dumps(record, ensure_ascii=False) + "\n")