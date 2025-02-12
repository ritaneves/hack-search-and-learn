import json
import yaml
from datasets import load_dataset
from copy import deepcopy
import sys


with open("precomputed_outputs_math_qa.json", "r") as f:
    precomputed_outputs = json.load(f)


def preparing_output_dataset(all_entries, responses, llm_name="llama", dataset_name="allenai/math_qa", split="test"):
    """
    {
        "input": {
            "prompt": "prompt 1", 
            "ideal_response": "response 1",
            "category": "",
            "source": ""
        },
        "response": "generated response 1",
        "llm_name": "llm name"
    },
    """

    all_outputs = []
    for entry in all_entries:
        if entry["prompt"] in responses.keys():
            all_outputs.append({"input": entry, "response": responses[entry["prompt"]], "llm_name": llm_name})

    with open(f"outputs.json", "w", encoding='utf8') as f:
        json.dump(all_outputs, f, ensure_ascii=False)


def preparing_input_dataset(dataset_name="allenai/math_qa", split="test"):
    """
    {
        "prompt":"prompt 1",
        "ideal_response": "ideal response 1",
        "category": "",
        "source": ""
    }
    """

    dataset = load_dataset(dataset_name, trust_remote_code=True)[split]
    
    all_entries = []

    i = 0
    for example in dataset:
        # Add to prompt and figure out unicode chars
        entry = {
            
                "prompt": example["Problem"] + "\n Options: " + example["options"].replace(" )", ".").replace(" ,", ","),
                "ideal_response": example["correct"],
                "category": dataset_name,
                "source": ""
                }
        i += 1
        if i > 99:
            break

        all_entries.append(entry)


    with open(f"inputs.json", "w", encoding='utf8') as f:
        json.dump(all_entries, f, ensure_ascii=False)

    return 

all_entries = preparing_input_dataset()
# preparing_output_dataset(all_entries, precomputed_outputs)