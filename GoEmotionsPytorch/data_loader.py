import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class InputExample(object):
    """ A single training/test example for simple sequence classification. """

    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(
        args,
        examples,
        tokenizer,
        max_length,
):
    processor = GoEmotionsProcessor(args)
    label_list_len = len(processor.get_labels())

    def convert_to_one_hot_label(label):
        one_hot_label = [0] * label_list_len
        for l in label:
            one_hot_label[l] = 1
        return one_hot_label

    labels = [convert_to_one_hot_label(example.label) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:10]):
        logger.info("*** Example ***")
        logger.info("guid: {}".format(example.guid))
        logger.info("sentence: {}".format(example.text_a))
        logger.info("tokens: {}".format(" ".join([str(x) for x in tokenizer.tokenize(example.text_a)])))
        logger.info("input_ids: {}".format(" ".join([str(x) for x in features[i].input_ids])))
        logger.info("attention_mask: {}".format(" ".join([str(x) for x in features[i].attention_mask])))
        logger.info("token_type_ids: {}".format(" ".join([str(x) for x in features[i].token_type_ids])))
        logger.info("label: {}".format(" ".join([str(x) for x in features[i].label])))

    return features


class GoEmotionsProcessor(object):
    """Processor for the GoEmotions data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        labels = []
        with open(os.path.join(os.path.dirname(__file__), self.args.data_dir, self.args.label_file), "r", encoding="utf-8") as f:
            for line in f:
                labels.append(line.rstrip())
        return labels

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            return f.readlines()

    def _create_examples(self, lines, set_type, df):
        """ Creates examples for the train, dev and test sets."""
        examples = []
        if (lines is not None):
            for (i, line) in enumerate(lines):
                guid = f"{set_type}-{i}"
                line = line.strip()
                items = line.split("\t")
                text_a = items[0]
                label = list(map(int, items[1].split(",")))
                if i % 5000 == 0:
                    logger.info(line)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        else:
            for (i, row) in df.iterrows():
                guid = "%s-%s" % (set_type, i)
                if isinstance(row["review_text"], float):
                    text_a = ""  # This accounts for rare encoding errors
                else:
                    text_a = row["review_text"]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=[]))
        return examples

    def get_examples(self, mode, df):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'pred':
            return self._create_examples(None, mode, df)
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file
        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.args.data_dir, file_to_read)), mode, df)

def load_and_cache_examples(args, tokenizer, mode, df):
    processor = GoEmotionsProcessor(args)
    if mode == "train":
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = processor.get_examples("train", None)
    elif mode == "dev":
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = processor.get_examples("dev", None)
    elif mode == "test":
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = processor.get_examples("test", None)
    elif mode == "pred":
        logger.info("Creating features from dataset")
        examples = processor.get_examples("pred", df)
    else:
        raise ValueError("For mode, only train, dev, test is available")
    features = convert_examples_to_features(
        args, examples, tokenizer, max_length=args.max_seq_len
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset, len(examples)
