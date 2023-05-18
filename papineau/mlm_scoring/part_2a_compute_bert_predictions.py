# Compute the probability of masculine, feminine, and gender-neutral variants
# of role nouns
# Adapted from:
# https://gist.github.com/yuchenlin/a2f42d3c4378ed7b83de65c7a2222eb2
# https://seaborn.pydata.org/generated/seaborn.kdeplot.html

import torch
import transformers
from transformers import BertForMaskedLM, BertTokenizer
import tqdm
import csv


MASK_INDEX = 103  # 103 corresponds to [MASK]

SPECIAL_TOKENS = [MASK_INDEX, 101, 102]

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


def mask_each_token(tokenized_sentences):
    # Mask each non-special token
    tokens = list(tokenized_sentences['input_ids'][0].numpy())
    input_ids = []
    for token_position, token_id in enumerate(tokenized_sentences['input_ids'][0]):
        if token_id not in SPECIAL_TOKENS:
            input_ids.append(tokens[:token_position] + [MASK_INDEX] + tokens[token_position + 1:])
    input_ids = torch.tensor(input_ids)

    return transformers.tokenization_utils_base.BatchEncoding({
        "input_ids": input_ids,
        "token_type_ids": tokenized_sentences['token_type_ids'].repeat(input_ids.shape[0], 1),
        "attention_mask": tokenized_sentences['attention_mask'].repeat(input_ids.shape[0], 1),
    })


def get_masked_indices(tokenized_sentences_masked):
    """Helper function for run_bert. Return a tensor sentence indices that contain mask, and a tensor of
    wordpiece indices that correspond to where mask was found.
    """
    masked_word_indices = torch.nonzero(
        tokenized_sentences_masked['input_ids'] == MASK_INDEX)

    # tensor of [[sentence_index, wordpiece_index]]
    return masked_word_indices[:, 0], masked_word_indices[:, 1]


def get_masked_probabilities(stimulus):

    # Tokenize stimulus sentence
    tokenized_sentences = tokenizer.batch_encode_plus(
        [stimulus], return_tensors='pt', add_special_tokens=True,
        padding=True, truncation=True)

    # Generate masked senetences, where each non-special token is masked
    tokenized_sentences_masked = mask_each_token(tokenized_sentences)
    tokenized_sentences_masked.to(device)

    # Find the index for the masked word
    masked_indices = get_masked_indices(tokenized_sentences_masked)

    # Get model logits for all words
    output = model(**tokenized_sentences_masked)
    logits = output["logits"]  # has shape [~6, ~8, 30522] for [n_sentences, max_sentence_length, vocab_size]

    # Extract the logits for the masked word, and convert to probabilities
    masked_logits = logits[masked_indices[0], masked_indices[1], :]
    masked_probabilities = torch.nn.functional.softmax(masked_logits, dim=1)

    # Return the probability of each masked token
    input_tokens = tokenizer.convert_ids_to_tokens(tokenized_sentences["input_ids"][0])
    result = []
    for i in range(masked_indices[0].shape[0]):
        masked_token_id = int(tokenized_sentences["input_ids"][torch.tensor(0), masked_indices[1][i]].numpy())
        result.append({
            "masked_token_id": masked_token_id,
            "masked_token_position": int(masked_indices[1][i].numpy()),
            "masked_token": tokenizer.convert_ids_to_tokens(masked_token_id),
            "stimulus_tokenized": input_tokens,
            "raw_probability": float(masked_probabilities[i, masked_token_id])
        })
    return result


def main(output_path="bert_predictions.csv"):
    input_sentences = load_masked_sentences()
    with open(output_path, "w") as f:
        csv_writer = csv.DictWriter(
            f, fieldnames=["stimulus", "name", "gender", "a/an", "role",
                           "role_gender", "state", "variants",
                           'masked_token_id', 'masked_token_position',
                           'masked_token', 'stimulus_tokenized',
                           'raw_probability'])
        csv_writer.writeheader()
        with torch.no_grad():  # This tells the model not to adjust its weights
            for i in tqdm.trange(len(input_sentences)):
                row = input_sentences[i]
                masked_probabilities_per_word = get_masked_probabilities(
                    row["stimulus"])
                for masked_row in masked_probabilities_per_word:
                    masked_row.update(row)
                    csv_writer.writerow(masked_row)


if __name__ == "__main__":
    main()
