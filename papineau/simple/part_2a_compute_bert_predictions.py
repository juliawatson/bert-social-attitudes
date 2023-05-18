# Compute the probability of masculine, feminine, and gender-neutral variants
# of role nouns
# Adapted from:
# https://gist.github.com/yuchenlin/a2f42d3c4378ed7b83de65c7a2222eb2
# https://seaborn.pydata.org/generated/seaborn.kdeplot.html

import torch
from transformers import BertForMaskedLM, BertTokenizer
import pandas as pd
import sys
import tqdm
import csv


MASK_INDEX = 103  # 103 corresponds to [MASK]



#### INITIALIZING THE MODEL ####

# The tokenizer splits sentences into word pieces (sometimes words
# are split into to multiple pieces), and maps word pieces to ids
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# The model takes in the ids and generates vector representation for each
# wordpiece.
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()  # This tells the model to behave in evaluation mode (not training mode)


# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


def load_masked_sentences(data_path="stimuli.csv"):
    result = []
    with open(data_path, "r") as f:
        dict_reader = csv.DictReader(f)
        for row in dict_reader:
            row["variants"] = eval(row["variants"])
            result.append(row)
    return result


def get_masked_variant_probabilities(stimulus, variants):
    # Tokenize stimulus sentence
    tokenized_sentences = tokenizer.batch_encode_plus(
        [stimulus], return_tensors='pt', add_special_tokens=True,
        padding=True, truncation=True)
    tokenized_sentences.to(device)

    # Find the index for the masked word
    masked_word_index_list = torch.nonzero(
        tokenized_sentences['input_ids'][0] == MASK_INDEX)
    assert masked_word_index_list.nelement() == 1, "There should be a masked token"
    masked_word_index = masked_word_index_list[0][0]

    # Get model logits for all words
    output = model(**tokenized_sentences)
    logits = output["logits"]  # has shape [1, ~8, 30522] for [n_sentences, max_sentence_length, vocab_size]

    # Extract the logits for the masked word, and convert to probabilities
    masked_logits = logits[0, masked_word_index, :]
    masked_probabilities = torch.nn.functional.softmax(masked_logits, dim=0)

    # Return the probability of each of the variants
    return {
        variant: float(masked_probabilities[tokenizer.vocab[variant]])
        for variant in variants}


def main(output_path="bert_predictions.csv"):
    input_sentences = load_masked_sentences()
    with open(output_path, "w") as f:
        csv_writer = csv.DictWriter(
            f, fieldnames=["stimulus", "name", "gender", "a/an", "masked_role",
                           "state", "variants", "variant_probabilities"])
        csv_writer.writeheader()
        with torch.no_grad():  # This tells the model not to adjust its weights
            for i in tqdm.trange(len(input_sentences)):
                row = input_sentences[i]
                probabilities = get_masked_variant_probabilities(
                    row["stimulus"], row["variants"])
                row["variant_probabilities"] = probabilities
                csv_writer.writerow(row)


if __name__ == "__main__":
    main()
