# coding=utf-8

import numpy as np
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
import jsonlines
import gensim
import tqdm
from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]


@dataclass(frozen=True)
class InputFeatures:
    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    passage_mask: Optional[List[int]]
    option_mask: Optional[List[int]]
    question_mask: Optional[List[int]]
    argument_bpe_ids: Optional[List[List[int]]]
    domain_bpe_ids: Optional[List[List[int]]]
    punct_bpe_ids: Optional[List[List[int]]]
    label: Optional[int]
    coocc: Optional[List[List[tuple]]]



class Split(Enum):
    train = "train"
    dev = "eval"
    test = "test"


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class MyMultipleChoiceDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            arg_tokenizer,
            relations,
            punctuations,
            task: str,
            max_seq_length: Optional[int] = None,
            max_ngram: int = 5,
            overwrite_cache=False,
            mode: Split = Split.train,
            demo=False,
        ):
            processor = processors[task]()

            if not os.path.isdir(os.path.join(data_dir, "cached_data")):
                os.mkdir(os.path.join(data_dir, "cached_data"))

            cached_features_file = os.path.join(
                data_dir,
                "cached_data",
                "dagn_cached_{}_{}_{}_{}_{}".format(
                    mode.value,
                    tokenizer.__class__.__name__,
                    str(max_seq_length),
                    task,
                    "_demo" if demo else ""
                ),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    label_list = processor.get_labels()
                    if mode == Split.dev:
                        if demo:
                            examples = processor.get_dev_demos(data_dir)
                        else:
                            examples = processor.get_dev_examples(data_dir)
                    elif mode == Split.test:
                        examples = processor.get_test_examples(data_dir)
                    elif mode == Split.train:
                        if demo:
                            examples = processor.get_train_demos(data_dir)
                        else:
                            examples = processor.get_train_examples(data_dir)
                    else:
                        raise Exception()
                    logger.info("Training examples: %s", len(examples))


                    self.features = convert_examples_to_arg_features(
                        examples,
                        label_list,
                        arg_tokenizer,
                        relations,
                        punctuations,
                        max_seq_length,
                        tokenizer,
                        max_ngram
                    )
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]



class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()



class ReclorProcessor(DataProcessor):
    """Processor for the ReClor data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train_qtype.json")), "train")

    def get_train_demos(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "100_train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "val_qtype.json")), "dev")

    def get_dev_demos(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "100_val.json")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test_qtype.json")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _read_jsonl(self, input_file):
        reader = jsonlines.Reader(open(input_file, "r"))
        lines = [each for each in reader]
        return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []
        for d in lines:
            context = d['context']
            question = d['question']
            answers = d['answers']
            label = 0 if type == "test" else d['label'] # for test set, there is no label. Just use 0 for convenience.
            id_string = d['id_string']
            # qtype = d['qtype']
            examples.append(
                InputExample(
                    example_id = id_string,
                    question = question,
                    contexts=[context, context, context, context],  # this is not efficient but convenient
                    endings=[answers[0], answers[1], answers[2], answers[3]],
                    label = label,
                    # qtype = [qtype, qtype, qtype, qtype]
                    )
                )
        return examples


class LogiQAProcessor(DataProcessor):
    """ Processor for the LogiQA data set. """

    def get_demo_examples(self, data_dir):
        logger.info("LOOKING AT {} demo".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "10_logiqa.txt")), "demo")

    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "Train.txt")), "train")

    def get_dev_examples(self, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "Eval.txt")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "Test.txt")), "test")

    def get_labels(self):
        return [0, 1, 2, 3]

    def _read_txt(self, input_file):
        with open(input_file, "r") as f:
            lines = f.readlines()
        return lines

    def _create_examples(self, lines, type):
        """ LogiQA: each 8 lines is one data point.
                The first line is blank line;
                The second is right choice;
                The third is context;
                The fourth is question;
                The remaining four lines are four options.
        """
        label_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        assert len(lines) % 8 ==0, 'len(lines)={}'.format(len(lines))
        n_examples = int(len(lines) / 8)
        examples = []
        # for i, line in enumerate(examples):
        for i in range(n_examples):
            label_str = lines[i*8+1].strip()
            context = lines[i*8+2].strip()
            question = lines[i*8+3].strip()
            answers = lines[i*8+4 : i*8+8]

            examples.append(
                InputExample(
                    example_id = " ",  # no example_id in LogiQA.
                    question = question,
                    contexts = [context, context, context, context],
                    endings = [item.strip()[2:] for item in answers],
                    label = label_map[label_str]
                )
            )
        assert len(examples) == n_examples
        return examples


def convert_examples_to_arg_features(
    examples: List[InputExample],
    label_list: List[str],
    arg_tokenizer,
    relations: Dict,
    punctuations: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    max_ngram: int,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`

    context -> chunks of context
            -> domain_words to Dids
    option -> chunk of option
           -> domain_words in Dids
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    cal_node = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            text_b = ending  # option

            if example.question.find("_") != -1:
                text_c = example.question.replace(" _", "")
            else:
                text_c = example.question

            stopwords = list(gensim.parsing.preprocessing.STOPWORDS) + punctuations
            inputs = arg_tokenizer(text_a, text_b, text_c, tokenizer, stopwords, relations, punctuations, max_ngram, max_length)
            # inputs["qtype"] = qtype
            choices_inputs.append(inputs)

        label = label_map[example.label]
        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )

        a_mask = [x["a_mask"] for x in choices_inputs]
        b_mask = [x["b_mask"] for x in choices_inputs]  # list[list]
        c_mask = [x["c_mask"] for x in choices_inputs]  # list[list]
        argument_bpe_ids = [x["argument_bpe_ids"] for x in choices_inputs]
        if isinstance(argument_bpe_ids[0], tuple):  # (argument_bpe_pattern_ids, argument_bpe_type_ids)
            arg_bpe_pattern_ids, arg_bpe_type_ids = [], []
            for choice_pattern, choice_type in argument_bpe_ids:
                assert (np.array(choice_pattern) > 0).tolist() == (np.array(choice_type) > 0).tolist(), 'pattern: {}\ntype: {}'.format(
                    choice_pattern, choice_type)
                arg_bpe_pattern_ids.append(choice_pattern)
                arg_bpe_type_ids.append(choice_type)
            argument_bpe_ids = (arg_bpe_pattern_ids, arg_bpe_type_ids)
        domain_bpe_ids = [x["domain_bpe_ids"] for x in choices_inputs]
        punct_bpe_ids = [x["punct_bpe_ids"] for x in choices_inputs]
        coocc_tags = [x["coocc"] for x in choices_inputs]

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=None,
                passage_mask=a_mask,
                option_mask=b_mask,
                question_mask=c_mask,
                argument_bpe_ids=argument_bpe_ids,
                domain_bpe_ids=domain_bpe_ids,
                punct_bpe_ids=punct_bpe_ids,
                label=label,
                coocc=coocc_tags,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features




processors = {"reclor": ReclorProcessor, "logiqa": LogiQAProcessor}
MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"reclor", 4, "logiqa", 4}

