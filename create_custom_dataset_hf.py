import pandas as pd
from datasets.dataset_dict import *
from datasets import *
import datasets
from PIL import Image
import json
from huggingface_hub import notebook_login


def preprocess_train(example):
    image = Image.open("E:\\INETUM\\INETUM_Datasets\\DocVQA\\rrc_docvqa\\rrc_docvqa\\train\\"+example["file_name"]).convert("RGB")
    new_image = {'image': image}
    return new_image

def preprocess_val(example):
    image = Image.open("E:\\INETUM\\INETUM_Datasets\\DocVQA\\rrc_docvqa\\rrc_docvqa\\val\\"+example["file_name"]).convert("RGB")
    new_image = {'image': image}
    return new_image


if __name__ == "__main__":
    LABELS_FILE_TRAIN = "E:\\INETUM\\INETUM_Datasets\\DocVQA\\rrc_docvqa\\rrc_docvqa\\train\\metadata_gt.csv"
    LABELS_FILE_VAL = "E:\\INETUM\\INETUM_Datasets\\DocVQA\\rrc_docvqa\\rrc_docvqa\\val\\metadata_gt.csv"

    df = pd.read_csv(LABELS_FILE_TRAIN, sep=";", engine="python")
    # display(df)
    dataset_train = Dataset.from_pandas(df).map(preprocess_train)
    # display(dataset_train)
    df = pd.read_csv(LABELS_FILE_VAL, sep=";", engine="python")
    # display(df)
    dataset_val = Dataset.from_pandas(df).map(preprocess_val)
    # display(dataset_val)

    example = dataset_train[0]
    json.loads(example['ground_truth'])
    example = dataset_val[0]
    json.loads(example['ground_truth'])

    test_image = dataset_train[0]["image"]
    # display(test_image)
    print("\nRESULT: ", dataset_train[0]["ground_truth"])
    # display
    print("\nDATASET FEATURES:")
    # display(dataset_train.features)
    print("\nDATASET ELEMENT:")
    test_image = dataset_val[0]["image"]
    # display(test_image)
    print("\nRESULT: ", dataset_val[0]["ground_truth"])
    # display
    print("\nDATASET FEATURES:")
    # display(dataset_val.features)
    print("\nDATASET ELEMENT:")

    dataset = DatasetDict({'train': dataset_train, 'test': dataset_val})
    # dataset


    notebook_login()
    ## note that you can push your dataset to the hub very easily (and reload afterwards using load_dataset)!
    dataset.push_to_hub("arvisioncode/donut-docvqa-base-12k", private=True)

