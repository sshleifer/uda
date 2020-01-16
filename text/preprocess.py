# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preprocessing for text classifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
# from absl import app
# from absl import flags

import numpy as np
import tensorflow as tf
from pathlib import Path

# from augmentation import aug_policy
from augmentation import sent_level_augment
from augmentation import word_level_augment
from utils import raw_data_utils
from utils import tokenization

from argparse import ArgumentParser

PARSER = ArgumentParser()

import logging

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler('tensorflow.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)


def create_arg(param_name, default, type, help='', required=False, **kwargs):
    PARSER.add_argument(f'--{param_name}', type=type, default=default, help=help, required=required,
                        **kwargs)





def get_data_for_worker(examples, replicas, worker_id):
    data_per_worker = len(examples) // replicas
    remainder = len(examples) - replicas * data_per_worker
    if worker_id < remainder:
        start = (data_per_worker + 1) * worker_id
        end = (data_per_worker + 1) * (worker_id + 1)
    else:
        start = data_per_worker * worker_id + remainder
        end = data_per_worker * (worker_id + 1) + remainder
    if worker_id == replicas - 1:
        assert end == len(examples)
    tf.compat.v1.logging.info("processing data from {:d} to {:d}".format(start, end))
    examples = examples[start: end]
    return examples, start, end


def build_vocab(examples):
    vocab = {}

    def add_to_vocab(word_list):
        for word in word_list:
            if word not in vocab:
                vocab[word] = len(vocab)

    for i in range(len(examples)):
        add_to_vocab(examples[i].word_list_a)
        if examples[i].text_b:
            add_to_vocab(examples[i].word_list_b)
    return vocab


def get_data_stats(data_stats_dir, sub_set, sup_size, replicas, examples):
    data_stats_dir = "{}/{}".format(data_stats_dir, sub_set)
    keys = ["tf_idf", "idf"]
    all_exist = True
    for key in keys:
        data_stats_path = "{}/{}.json".format(data_stats_dir, key)
        if not tf.io.gfile.Exists(data_stats_path):
            all_exist = False
            tf.compat.v1.logging.info("Not exist: {}".format(data_stats_path))
    if all_exist:
        tf.compat.v1.logging.info("loading data stats from {:s}".format(data_stats_dir))
        data_stats = {}
        for key in keys:
            with tf.io.gfile.Open(
                    "{}/{}.json".format(data_stats_dir, key)) as inf:
                data_stats[key] = json.load(inf)
    else:
        assert sup_size == -1, "should use the complete set to get tf_idf"
        assert replicas == 1, "should use the complete set to get tf_idf"
        data_stats = word_level_augment.get_data_stats(examples)
        tf.io.gfile.makedirs(data_stats_dir)
        for key in keys:
            with tf.io.gfile.Open("{}/{}.json".format(data_stats_dir, key), "w") as ouf:
                json.dump(data_stats[key], ouf)
        tf.compat.v1.logging.info("dumped data stats to {:s}".format(data_stats_dir))
    return data_stats


def tokenize_examples(examples, tokenizer):
    tf.compat.v1.logging.info("tokenizing examples")
    for i in range(len(examples)):
        examples[i].word_list_a = tokenizer.tokenize_to_word(examples[i].text_a)
        if examples[i].text_b:
            examples[i].word_list_b = tokenizer.tokenize_to_word(examples[i].text_b)
        if i % 10000 == 0:
            tf.compat.v1.logging.info("finished tokenizing example {:d}".format(i))
    return examples


def convert_examples_to_features(
        examples, label_list, seq_length, tokenizer, trunc_keep_right,
        data_stats=None, aug_ops=None):
    """convert examples to features."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tf.compat.v1.logging.info("number of examples to process: {}".format(len(examples)))

    features = []

    if aug_ops:
        tf.compat.v1.logging.info("building vocab")
        word_vocab = build_vocab(examples)
        examples = word_level_augment.word_level_augment(
            examples, aug_ops, word_vocab, data_stats
        )

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.compat.v1.logging.info("processing {:d}".format(ex_index))
        tokens_a = tokenizer.tokenize_to_wordpiece(example.word_list_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize_to_wordpiece(example.word_list_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            if trunc_keep_right:
                _truncate_seq_pair_keep_right(tokens_a, tokens_b, seq_length - 3)
            else:
                _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                if trunc_keep_right:
                    tokens_a = tokens_a[-(seq_length - 2):]
                else:
                    tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        label_id = label_map[example.label]
        if ex_index < 1:
            tf.compat.v1.logging.info("*** Example ***")
            tf.compat.v1.logging.info("guid: %s" % (example.guid))
            # st = " ".join([str(x) for x in tokens])
            st = ""
            for x in tokens:
                st += str(x) + " "
            tf.compat.v1.logging.info("tokens: %s" % st)
            tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.compat.v1.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.compat.v1.logging.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
            tf.compat.v1.logging.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                label_id=label_id))
    return features


def _create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_type_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.label_id = label_id

    def get_dict_features(self):
        return {
            "input_ids": _create_int_feature(self.input_ids),
            "input_mask": _create_int_feature(self.input_mask),
            "input_type_ids": _create_int_feature(self.input_type_ids),
            "label_ids": _create_int_feature([self.label_id])
        }


class PairedUnsupInputFeatures(object):
    """Features for paired unsup data."""

    def __init__(self, ori_input_ids, ori_input_mask, ori_input_type_ids,
                 aug_input_ids, aug_input_mask, aug_input_type_ids):
        self.ori_input_ids = ori_input_ids
        self.ori_input_mask = ori_input_mask
        self.ori_input_type_ids = ori_input_type_ids
        self.aug_input_ids = aug_input_ids
        self.aug_input_mask = aug_input_mask
        self.aug_input_type_ids = aug_input_type_ids

    def get_dict_features(self):
        return {
            "ori_input_ids": _create_int_feature(self.ori_input_ids),
            "ori_input_mask": _create_int_feature(self.ori_input_mask),
            "ori_input_type_ids": _create_int_feature(self.ori_input_type_ids),
            "aug_input_ids": _create_int_feature(self.aug_input_ids),
            "aug_input_mask": _create_int_feature(self.aug_input_mask),
            "aug_input_type_ids": _create_int_feature(self.aug_input_type_ids),
        }


def obtain_tfrecord_writer(data_path, worker_id, shard_cnt):
    tfrecord_writer = tf.io.TFRecordWriter(
        os.path.join(
            data_path,
            "tf_examples.tfrecord.{:d}.{:d}".format(worker_id, shard_cnt)))
    return tfrecord_writer


def dump_tfrecord(features, data_path, worker_id=None, max_shard_size=4096):
    """Dump tf record."""
    if not Path(data_path).exists():
        tf.io.gfile.makedirs(data_path)
    tf.compat.v1.logging.info("dumping TFRecords")
    np.random.shuffle(features)
    shard_cnt = 0
    shard_size = 0
    tfrecord_writer = obtain_tfrecord_writer(data_path, worker_id, shard_cnt)
    for feature in features:
        tf_example = tf.train.Example(
            features=tf.train.Features(feature=feature.get_dict_features()))
        if shard_size >= max_shard_size:
            tfrecord_writer.close()
            shard_cnt += 1
            tfrecord_writer = obtain_tfrecord_writer(data_path, worker_id, shard_cnt)
            shard_size = 0
        shard_size += 1
        tfrecord_writer.write(tf_example.SerializeToString())
    tfrecord_writer.close()


def get_data_by_size_lim(train_examples, processor, sup_size):
    """Deterministicly get a dataset with only sup_size examples."""
    # Assuming sup_size < number of labeled data and
    # that there are same number of examples for each category
    assert sup_size % len(processor.get_labels()) == 0
    per_label_size = sup_size // len(processor.get_labels())
    per_label_examples = {}
    for i in range(len(train_examples)):
        label = train_examples[i].label
        if label not in per_label_examples:
            per_label_examples[label] = []
        per_label_examples[label] += [train_examples[i]]

    for label in processor.get_labels():
        assert len(per_label_examples[label]) >= per_label_size, (
            "label {} only has {} examples while the limit"
            "is {}".format(label, len(per_label_examples[label]), per_label_size))

    new_train_examples = []
    for i in range(per_label_size):
        for label in processor.get_labels():
            new_train_examples += [per_label_examples[label][i]]
    train_examples = new_train_examples
    return train_examples


def _truncate_seq_pair_keep_right(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop(0)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def proc_and_save_sup_data(
        processor, sub_set, raw_data_dir, sup_out_dir,
        tokenizer, max_seq_length, trunc_keep_right,
        worker_id, replicas, sup_size):
    tf.compat.v1.logging.info("getting examples")
    if sub_set == "train":
        examples = processor.get_train_examples(raw_data_dir)
    elif sub_set == "dev":
        examples = processor.get_dev_examples(raw_data_dir)
        assert replicas == 1, "dev set can be processsed with just one worker"
        assert sup_size == -1, "should use the full dev set"

    if sup_size != -1:
        tf.compat.v1.logging.info("setting number of examples to {:d}".format(
            sup_size))
        examples = get_data_by_size_lim(
            examples, processor, sup_size)
    if replicas != 1:
        if len(examples) < replicas:
            replicas = len(examples)
            if worker_id >= replicas:
                return
        examples = get_data_for_worker(
            examples, replicas, worker_id)

    tf.compat.v1.logging.info("processing data")
    examples = tokenize_examples(examples, tokenizer)

    features = convert_examples_to_features(
        examples, processor.get_labels(), max_seq_length, tokenizer,
        trunc_keep_right, None, None)
    dump_tfrecord(features, sup_out_dir, worker_id)


def proc_and_save_unsup_data(
        processor, sub_set,
        raw_data_dir, data_stats_dir, unsup_out_dir,
        tokenizer,
        max_seq_length, trunc_keep_right,
        aug_ops, aug_copy_num,
        worker_id, replicas):
    # print random seed just to double check that we use different random seeds
    # for different runs so that we generate different augmented examples for the same original
    # example.
    random_seed = np.random.randint(0, 100000)
    tf.compat.v1.logging.info("random seed: {:d}".format(random_seed))
    np.random.seed(random_seed)
    tf.compat.v1.logging.info("getting examples")

    if sub_set == "train":
        ori_examples = processor.get_train_examples(raw_data_dir)
    elif sub_set.startswith("unsup"):
        ori_examples = processor.get_unsup_examples(raw_data_dir, sub_set)
    else:
        assert False
    # this is the size before spliting data for each worker
    data_total_size = len(ori_examples)
    if replicas != -1:
        ori_examples, start, end = get_data_for_worker(
            ori_examples, replicas, worker_id)
    else:
        start = 0
        end = len(ori_examples)

    tf.compat.v1.logging.info("getting augmented examples")
    aug_examples = copy.deepcopy(ori_examples)
    aug_examples = sent_level_augment.run_augment(
        aug_examples, aug_ops, sub_set,
        aug_copy_num,
        start, end, data_total_size)

    labels = processor.get_labels() + ["unsup"]
    tf.compat.v1.logging.info("processing ori examples")
    ori_examples = tokenize_examples(ori_examples, tokenizer)
    ori_features = convert_examples_to_features(
        ori_examples, labels, max_seq_length, tokenizer,
        trunc_keep_right, None, None)

    if "idf" in aug_ops:
        data_stats = get_data_stats(
            data_stats_dir, sub_set,
            -1, replicas, ori_examples)
    else:
        data_stats = None

    tf.compat.v1.logging.info("processing aug examples")
    aug_examples = tokenize_examples(aug_examples, tokenizer)
    aug_features = convert_examples_to_features(
        aug_examples, labels, max_seq_length, tokenizer,
        trunc_keep_right, data_stats, aug_ops)

    unsup_features = []
    for ori_feat, aug_feat in zip(ori_features, aug_features):
        unsup_features.append(PairedUnsupInputFeatures(
            ori_feat.input_ids,
            ori_feat.input_mask,
            ori_feat.input_type_ids,
            aug_feat.input_ids,
            aug_feat.input_mask,
            aug_feat.input_type_ids,
        ))
    dump_tfrecord(unsup_features, unsup_out_dir, worker_id)


def main(args):
    if args.max_seq_length > 512:
        raise ValueError(
            "Cannot use sequence length {:d} because the BERT model "
            "was only trained up to sequence length {:d}".format(
                args.max_seq_length, 512))

    processor = raw_data_utils.get_processor(args.task_name)
    # Create tokenizer
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    if args.data_type == "sup":
        sup_out_dir = args.output_base_dir
        tf.compat.v1.logging.info("Create sup. data: subset {} => {}".format(
            args.sub_set, sup_out_dir))

        proc_and_save_sup_data(
            processor, args.sub_set, args.raw_data_dir, sup_out_dir,
            tokenizer, args.max_seq_length, args.trunc_keep_right,
            args.worker_id, args.replicas, args.sup_size,
        )
    elif args.data_type == "unsup":
        assert args.aug_ops is not None, \
            "aug_ops is required to preprocess unsupervised data."
        unsup_out_dir = os.path.join(
            args.output_base_dir,
            args.aug_ops,
            str(args.aug_copy_num))
        data_stats_dir = os.path.join(args.raw_data_dir, "data_stats")

        tf.compat.v1.logging.info("Create unsup. data: subset {} => {}".format(
            args.sub_set, unsup_out_dir))
        proc_and_save_unsup_data(
            processor, args.sub_set,
            args.raw_data_dir, data_stats_dir, unsup_out_dir,
            tokenizer, args.max_seq_length, args.trunc_keep_right,
            args.aug_ops, args.aug_copy_num,
            args.worker_id, args.replicas)


if __name__ == "__main__":
    #PARSER.add_argument("task_name", default="IMDB", type=str, help="The name of the task to train.")
    create_arg("task_name", default="IMDB", type=str, help="The name of the task to train.")
    create_arg("raw_data_dir", None, str, "Data directory of the raw data")
    create_arg("output_base_dir", None, str, "Data directory of the processed data")
    create_arg("aug_ops", "bt-0.9", str, "augmentation method")
    create_arg("aug_copy_num", -1, int,
               help="We generate multiple augmented examples for oneunlabeled example, "
                    "aug_copy_num "
                    "is the index of the generated augmentedexample")
    create_arg("max_seq_length", 512, int,
               help="The maximum total sequence length after WordPiece tokenization. Sequences "
                    "longer "
                    "than this will be truncated, and sequences shorter than this will be padded.")
    create_arg("sup_size", -1, int, "size of the labeled set")
    create_arg("trunc_keep_right", True, bool,
               help="Whether to keep the right part when truncate a sentence.")
    create_arg("data_type", "sup", str, help="Which preprocess task to perform.",
               choices=["sup", "unsup"])
    create_arg("sub_set", "train", str,
               "Which sub_set to preprocess. The sub_set can be train, dev and unsup_in")
    create_arg("vocab_file", "", str, "The path of the vocab file of BERT.")
    create_arg("do_lower_case", True, bool, "Whether to use uncased text for BERT.")
    create_arg("back_translation_dir", "", str, "Directory for back translated sentence.")
    create_arg("replicas", 1, int,
               "An argument for parallel preprocessing. For example, when replicas=3,we divide the "
               "data into three parts, and only process one part according to the worker_id.")
    create_arg("worker_id", 0, int,
               "An argument for parallel preprocessing. See 'replicas' for more details")
    args = PARSER.parse_args()
    main(args)
