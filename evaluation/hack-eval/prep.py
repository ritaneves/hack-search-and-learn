import json
import yaml
from datasets import load_dataset
from copy import deepcopy
import sys
import re


def match_qa(x, choices):

    convert = ["a", "b", "c", "d", "e"]

    choices = choices.split(",")
    choices = [x.split(") ")[-1].strip() for x in choices]

    match = re.search(r"\\boxed{([^}]*)}", x)
    if match:
        answer = match.group(1)
        try: 
            idx = choices.index(answer)
            return convert[idx]
        except:
            return "x"
    else:
        return "x"


def preparing_output_dataset(all_entries, all_options, responses, dataset_name="allenai/math_qa", split="test", llm_name="llama", method="best_of_n"):
    """
    {
        "input": {
            "prompt": "prompt 1", 
            "ideal_response": "response 1",
            "category": "",
            "source": "",
            "method": search method
        },
        "response": "generated response 1",
        "llm_name": "llm name"
    },
    """

    all_outputs = []
    for i, entry in enumerate(all_entries):
        if entry["prompt"] in responses.keys():
            answer = match_qa(responses[entry["prompt"]], all_options[i]) 
        
            all_outputs.append({"input": entry, "response": answer, "llm_name": llm_name})

    with open(f"{method}_outputs.json", "w", encoding='utf8') as f:
        json.dump(all_outputs, f, ensure_ascii=False)

    return


def preparing_input_dataset(dataset_name="allenai/math_qa", split="test", limit=100):
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
    all_options = []

    i = 0
    for example in dataset:
        # Add to prompt and figure out unicode chars
        entry = {
                "prompt": example["Problem"], # + "\n Options: " + example["options"].replace(" )", ".").replace(" ,", ","),
                "ideal_response": example["correct"],
                "category": dataset_name,
                "source": ""
                }
        i += 1
        if i > limit:
            break

        all_entries.append(entry)
        all_options.append(example["options"])

    return all_entries, all_options