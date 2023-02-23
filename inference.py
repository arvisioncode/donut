import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from donut import DonutModel, JSONParseEvaluator, load_json, save_json


def test(args):
    pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)
    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")
    pretrained_model.eval()

    question = f"<s_docvqa><s_question>{args.question.lower()}</s_question><s_answer>"
    img = Image.open(args.image_path)
    print(f"INPUT:: image:{print(args.image_path)} question:{question}")
    
    output = pretrained_model.inference(image=img,prompt=question,)["predictions"][0]
    print(f"OUTPUT:: answer:{output}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--question", type=str, default=None)
    args, left_argv = parser.parse_known_args()

    print("TASK NAME: docvqa")
    predictions = test(args)
