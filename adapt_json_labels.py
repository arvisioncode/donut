import json
import re
from datasets import load_dataset
import pandas as pd


def add_ground_truth(examples):
    question = examples['question']
    answers = examples['answers']
    ground_truths = []

    # question = question.replace("\\", "") # this was just one corrupt example (index 91 of training set)
    query = re.sub(' +', ' ', question)
    query = query.replace('"', '\\"')
    # let's create the ground truth string
    ground_truth_example = '{"gt_parses": ['
    for idx, answer in enumerate(answers):
        answer = answer.replace('"', '\\"')
        ground_truth_example += '{"question" : "' + query + '", "answer" : "' + answer + '"}'
        # add comma
        # print("idx: ", idx)
        # print("len(answers): ", len(answers))
        if idx < len(answers) - 1:
            ground_truth_example += ', '
    ground_truth_example += ']}'
    # ground_truths.append(ground_truth_example)
    # examples['ground_truth'] = ground_truths
    examples['ground_truth'] = ground_truth_example
    # print(examples)
    return examples


def add_file_name(examples):
    reult = []
    # path = "{" + examples["image"] + "}"
    path = examples['image']
    # reult.append(path)
    # examples['file_name'] = reult
    examples['file_name'] = path
    # print(examples['file_name'])
    return examples


def main_json(FILE_INPUT, FILE_OUTPUT):
    input = open(FILE_INPUT)
    output = open(FILE_OUTPUT, 'w') 
    data = json.load(input)

    for line in data:
        line = add_file_name(line)
        line = add_ground_truth(line)

        line_out = line
        del line_out["image"]
        del line_out["question"]
        del line_out["docId"]
        del line_out["answers"]
        del line_out["data_split"]
        del line_out["is_table"]

        # print(line_out)
        json.dump(line_out, output)
        output.write('\n')

    input.close()
    output.close()


def main_excell(FILE_INPUT, FILE_OUTPUT):
    input_file = open(FILE_INPUT)
    data = json.load(input_file)

    # Create a list to hold the processed data
    line_result = []

    # Loop through the data and process each line
    for line in data:
        line = add_file_name(line)
        line = add_ground_truth(line)

        # Remove unnecessary fields
        del line["image"]
        del line["question"]
        del line["docId"]
        del line["answers"]
        del line["data_split"]
        del line["is_table"]

        print(line)
        line_result.append(line)

    # Create a DataFrame from the processed data
    df = pd.DataFrame(line_result)
    # Save the DataFrame to an Excel file
    df.to_csv(FILE_OUTPUT, sep=';', index=False)
    print(df)
    input_file.close()


if __name__ == "__main__":
    FILE_INPUT = "./dataset/bupa_docvqa_dataset_v2/metadata.json"
    FILE_OUTPUT = "./dataset/bupa_docvqa_dataset_v2/metadata_gt.jsonl"
    main_json(FILE_INPUT, FILE_OUTPUT)

    FILE_OUTPUT = "./dataset/bupa_docvqa_dataset_v2/metadata_gt.csv"
    main_excell(FILE_INPUT, FILE_OUTPUT)
    ### IMPORTANTE: REEMPLAZAR A MANO "" POR "
    

    # Check dataset created as will be downloaded in training
    data = load_dataset("dataset/bupa_docvqa_dataset_v2/", split="train")
    # print(data)
    # print(data[10])