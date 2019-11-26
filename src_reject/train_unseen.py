import os
import sys
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import math
import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorlayer as tl
from datetime import datetime, timedelta
from random import randint
import progressbar
import nltk
from scipy.spatial.distance import cosine

import src_reject.log as log
import src_reject.utils as utils
import src_reject.error as error
import src_reject.config as config
import src_reject.model_unseen as model_unseen
import src_reject.train_base as train_base
import src_reject.dataloader as dataloader

results_path = "../results/"


class Controller4Unseen(train_base.Base_Controller):

    def __init__(
            self,
            model,
            vocab,
            class_dict,
            kg_vector_dict,
            word_embed_mat,
            gpu_config=None,
            random_unseen_class=False,
            random_percentage=0.25,
            random_unseen_class_list=None,
            base_epoch=-1,
            lemma=False
    ):
        super(Controller4Unseen, self).__init__(model, gpu_config)

        logging = log.Log(sys.stdout, self.log_save_dir + "log-%s" % utils.now2string())
        sys.stdout = logging

        self.base_epoch = base_epoch
        self.vocab = vocab
        self.class_dict = class_dict
        self.kg_vector_dict = kg_vector_dict
        self.word_embed_mat = word_embed_mat
        self.lemma = lemma

        if self.lemma:
            from nltk.stem import WordNetLemmatizer
            self.lemmatizer = WordNetLemmatizer

        if random_unseen_class:
            num_class = len(class_dict.keys())
            num_unseen_class = int(num_class * random_percentage)
            class_id_list = list(class_dict.keys())
            random.shuffle(class_id_list)
            self.unseen_class = class_id_list[:num_unseen_class]
            print("Random selected unseen class %s" % (self.unseen_class))
        else:
            assert random_unseen_class_list is not None
            self.unseen_class = random_unseen_class_list
            print("Assigned unseen class %s" % (self.unseen_class))

        self.seen_class = list()
        for key in self.class_dict.keys():
            if key not in self.unseen_class:
                self.seen_class.append(key)
        print("Seen class %s" % (self.seen_class))

    def check_seen(self, class_id):
        return not class_id in self.unseen_class

    def get_text_of_seen_class(self, text_seqs, class_list):
        print("Getting text of seen classes")
        assert len(text_seqs) == len(class_list)
        seen_text_seqs = list()
        seen_class_list = list()

        with progressbar.ProgressBar(max_value=len(text_seqs)) as bar:
            for idx, text in enumerate(text_seqs):
                if self.check_seen(class_list[idx]):
                    seen_text_seqs.append(text)
                    seen_class_list.append(class_list[idx])
                bar.update(idx)
        assert len(seen_text_seqs) == len(seen_class_list)
        print("Text seqs of seen classes: %d" % len(seen_text_seqs))
        return seen_text_seqs, seen_class_list

    def get_text_of_unseen_class(self, text_seqs, class_list, augdata=False):
        print("Getting text of unseen classes", len(text_seqs))
        assert len(text_seqs) == len(class_list)
        unseen_text_seqs = list()
        unseen_class_list = list()

        unseen_class_counter = dict()
        for c in self.unseen_class:
            unseen_class_counter[c] = 0

        with progressbar.ProgressBar(max_value=len(text_seqs)) as bar:
            for idx, text in enumerate(text_seqs):
                if not self.check_seen(class_list[idx]):

                    # control the number of augmented data
                    if augdata and unseen_class_counter[class_list[idx]] >= config.augmentation:
                        continue

                    unseen_class_counter[class_list[idx]] += 1

                    unseen_text_seqs.append(text)
                    unseen_class_list.append(class_list[idx])
                bar.update(idx)
        assert len(unseen_text_seqs) == len(unseen_class_list)
        print("Text seqs of unseen classes: %d" % len(unseen_text_seqs))
        if augdata:
            print("Augmented data num of each unseen class {%s}" % (
                ", ".join(["%s:%s" % (k, v) for k, v in unseen_class_counter.items()])))
        print("unseen_text_seqs", len(unseen_text_seqs), unseen_class_list)
        return unseen_text_seqs, unseen_class_list

    def prepro_encode(self, textlist):
        newtextlist = list()
        for idx, text in enumerate(textlist):
            # if len(text[1:-1]) > self.model.max_length:
            #     startid = 1 + randint(0, len(text[1:-1]) - self.model.max_length)
            # else:
            #     startid = 1
            startid = 1
            newtextlist.append(text[startid:-1] + [self.vocab.pad_id])
        newtextlist = tl.prepro.pad_sequences(newtextlist, maxlen=self.model.max_length, dtype='int64', padding='post',
                                              truncating='post', value=self.vocab.pad_id)
        for idx, text in enumerate(newtextlist):
            newtextlist[idx] = text[:-1] + [self.vocab.pad_id]

        text_array = np.zeros((len(newtextlist), self.model.max_length, self.model.word_embedding_dim))
        for idx, text in enumerate(newtextlist):
            for widx, word_id in enumerate(text):
                text_array[idx][widx] = self.word_embed_mat[word_id]

        return np.array(newtextlist), text_array

    def get_random_text(self, num):
        random_text = list()
        for i in range(num):
            text = [randint(0, self.vocab.unk_id) for _ in range(self.model.max_length)]
            random_text.append(text)
        return random_text

    def get_random_class(self, true_class_id_list, category_logits):
        class_id_list = list()

        for idx, logit in enumerate(category_logits):
            true_class_id = true_class_id_list[idx]
            if config.augmentation == 0:
                assert self.check_seen(true_class_id)
            if logit == 1:
                class_id_list.append(true_class_id)
            elif logit == 0:
                while True:
                    random_class_id = random.choice(list(self.class_dict))
                    # random choice class must be seen class
                    if random_class_id != true_class_id and self.check_seen(random_class_id):
                        break
                class_id_list.append(random_class_id)
            else:
                raise ValueError("category logit has to be 0 or 1.")
        return np.array(class_id_list)

    def get_class_label_embed(self, class_id_list):
        class_embed = np.zeros((len(class_id_list), self.model.max_length, self.model.word_embedding_dim))

        for idx, class_id in enumerate(class_id_list):
            class_embed[idx, :, :] = np.repeat([self.word_embed_mat[self.vocab.word_to_id(self.class_dict[class_id])]],
                                               self.model.max_length, axis=0)

        return class_embed

    def get_kg_vector_given_class(self, encode_text_seqs, class_id_list):

        # TODO: remove to add kg_vector for training
        # return np.zeros((config.batch_size, self.model.max_length, config.kg_embedding_dim))

        assert encode_text_seqs.shape[0] == class_id_list.shape[0]

        kg_vector_list = list()

        for idx, class_id in enumerate(class_id_list):

            kg_vector = np.zeros([self.model.max_length, config.kg_embedding_dim])

            for widx, word_id in enumerate(encode_text_seqs[idx]):
                word = self.vocab.id_to_word(word_id)
                if self.lemma:
                    new_word = nltk.pos_tag([word])  # a list of words à a list of words with part of speech
                    new_word = [self.lemmatizer.lemmatize(t[0], config.pos_dict[t[1]]) for t in new_word if
                                t[1] in config.pos_dict]
                    if len(new_word) > 0:
                        word = new_word[0]
                kg_vector[widx, :] = dataloader.get_kg_vector(self.kg_vector_dict, self.class_dict[class_id], word)

            kg_vector_list.append(kg_vector)

        kg_vector_list = np.array(kg_vector_list)

        assert kg_vector_list.shape == (config.batch_size, self.model.max_length, config.kg_embedding_dim)

        return kg_vector_list

    def get_pred_class_topk(self, pred_mat, k=1):
        assert k > 0
        pred_k = list()
        for i in range(pred_mat.shape[0]):
            confidence = pred_mat[i]
            topk = np.argsort(confidence)[-k:] + 1
            for class_id in topk:
                assert class_id in self.class_dict
            pred_k.append(topk)
        pred_k = np.array(pred_k)
        pred_k = np.reshape(pred_k, newshape=(pred_mat.shape[0], k))
        assert pred_k.shape == (pred_mat.shape[0], k)
        return pred_k

    def get_one_hot_results(self, class_list):
        one_hot_res = np.zeros(shape=(class_list.shape[0], len(self.class_dict)))
        for idx, item in enumerate(class_list):
            for class_id in item:
                assert class_id in self.class_dict
                one_hot_res[idx, class_id - 1] = 1
        return one_hot_res

    def __train__(self, epoch, text_seqs, class_list, max_train_steps=None):

        assert len(text_seqs) == len(class_list)

        # train_order = list(range(len(text_seqs)))
        # random.shuffle(train_order)

        start_time = time.time()
        step_time = time.time()

        all_loss = np.zeros(1)

        train_steps = len(text_seqs) // config.batch_size

        if max_train_steps is not None and max_train_steps < train_steps:
            train_steps = max_train_steps

        train_order = random.sample(range(len(text_seqs)), k=train_steps * config.batch_size)

        for cstep in range(train_steps):
            global_step = cstep + epoch * train_steps

            # category_logits = [1 if randint(0, config.negative_sample) == 0 else 0 for _ in range(config.batch_size)]
            category_logits = [1 if randint(0, config.negative_sample + epoch * config.negative_increase) == 0 else 0
                               for _ in range(config.batch_size)]
            # category_logits = [1 if randint(0, config.negative_sample + epoch * 3) == 0 else 0 for _ in range(config.batch_size)]

            true_class_id_mini = [class_list[idx] for idx in
                                  train_order[cstep * config.batch_size: (cstep + 1) * config.batch_size]]
            text_seqs_mini = [text_seqs[idx] for idx in
                              train_order[cstep * config.batch_size: (cstep + 1) * config.batch_size]]

            if not config.model == "autoencoder":
                # random text
                true_class_id_mini = true_class_id_mini[:-3] + [-1, -1, -1]
                text_seqs_mini = text_seqs_mini[:-3] + self.get_random_text(3)
            else:
                tmpid = random.choice(list(self.class_dict.keys()))
                text_seqs_mini = text_seqs_mini[:-1] + \
                                 [[self.vocab.start_id, self.vocab.word_to_id(self.class_dict[tmpid]),
                                   self.vocab.end_id]]

            encode_seqs_id_mini, encode_seqs_mat_mini = self.prepro_encode(text_seqs_mini)

            # for class_id in seen_class_list:

            category_logits = category_logits[:-3] + [0, 0, 0]
            class_id_mini = self.get_random_class(true_class_id_mini, category_logits)
            # class_id_mini = np.array([class_id] * config.batch_size)
            # category_logits = [int(class_id_mini[i] == true_class_id_mini[i]) for i in range(config.batch_size)]

            class_label_embed_mini = self.get_class_label_embed(class_id_mini)

            if config.model == "cnnfc" or config.model == "rnnfc":
                results = self.sess.run([
                    self.model.train_loss,
                    self.model.train_net.outputs,
                    self.model.learning_rate,
                    self.model.optim
                ], feed_dict={
                    self.model.encode_seqs: encode_seqs_mat_mini,
                    self.model.class_label_seqs: class_label_embed_mini,
                    self.model.category_logits: np.expand_dims(np.array(category_logits), -1),
                    self.model.global_step: global_step,
                })
            elif config.model == "autoencoder":
                decode_seqs_mat_mini = np.concatenate(
                    (np.zeros(shape=(encode_seqs_mat_mini.shape[0], 1, encode_seqs_mat_mini.shape[2])),
                     encode_seqs_mat_mini[:, :-1, :]),
                    axis=1
                )
                results = self.sess.run([
                    self.model.train_loss,
                    self.model.train_net.outputs,
                    self.model.learning_rate,
                    self.model.optim
                ], feed_dict={
                    self.model.encode_seqs: encode_seqs_mat_mini,
                    self.model.decode_seqs: decode_seqs_mat_mini,
                    self.model.global_step: global_step,
                    self.model.target_seqs: encode_seqs_id_mini,
                    self.model.target_mask: np.where(encode_seqs_id_mini == self.vocab.pad_id, 0, 1)
                })
            else:
                if config.model == "vwvc":
                    kg_vector_seqs_mini = np.zeros(
                        (config.batch_size, self.model.max_length, self.model.kg_embedding_dim))
                else:
                    kg_vector_seqs_mini = self.get_kg_vector_given_class(encode_seqs_id_mini, class_id_mini)
                results = self.sess.run([
                    self.model.train_loss,
                    self.model.train_net.outputs,
                    # self.model.train_align.outputs,
                    self.model.learning_rate,
                    self.model.optim
                ], feed_dict={
                    self.model.encode_seqs: encode_seqs_mat_mini,
                    self.model.class_label_seqs: class_label_embed_mini,
                    self.model.kg_vector: kg_vector_seqs_mini,
                    self.model.category_logits: np.expand_dims(np.array(category_logits), -1),
                    self.model.global_step: global_step,
                })

            all_loss += results[:1]

            if cstep % config.cstep_print_unseen == 0 and cstep > 0:
                print(
                    "[Train] Epoch: [%3d][%4d/%4d] time: %.4f, lr: %.8f, loss: %s" %
                    (epoch, cstep, train_steps, time.time() - step_time, results[-2], all_loss / (cstep + 1))
                )
                step_time = time.time()
                # print(time.time() - step_time)
                # exit()

        print(
            "[Train Sum] Epoch: [%3d] time: %.4f, lr: %.8f, loss: %s" %
            (epoch, time.time() - start_time, results[-2], all_loss / train_steps)
        )

        return all_loss / train_steps

    def __test_autoencoder__(self, epoch, text_seqs, class_list):
        assert config.model == "autoencoder"

        start_time = time.time()
        step_time = time.time()

        test_steps = len(text_seqs) // config.batch_size

        topk_list = list()
        pred_class_list = list()

        class_id_mini = np.array(list(self.class_dict.keys()) + [1] * (config.batch_size - len(self.class_dict)))
        class_label_embed_mini = self.get_class_label_embed(class_id_mini)

        class_text_state = self.sess.run([
            self.model.test_text_state
        ], feed_dict={
            self.model.encode_seqs: class_label_embed_mini,
        })
        class_text_state = class_text_state[0][0]

        class_text_state_dict = dict()
        for cidx, class_id in enumerate(list(class_id_mini)):
            if class_id not in class_text_state_dict:
                class_text_state_dict[class_id] = class_text_state[cidx]

        for cstep in range(test_steps):

            text_seqs_mini = text_seqs[cstep * config.batch_size: (cstep + 1) * config.batch_size]

            encode_seqs_id_mini, encode_seqs_mat_mini = self.prepro_encode(text_seqs_mini)

            pred_mat = np.zeros([config.batch_size, len(self.class_dict)])

            test_text_state = self.sess.run([
                self.model.test_text_state,
            ], feed_dict={
                self.model.encode_seqs: encode_seqs_mat_mini,
            })
            test_text_state = test_text_state[0][0]

            for ti in range(config.batch_size):
                pc = None
                sim = None
                for class_id in self.class_dict:
                    cossim = 1 - cosine(class_text_state_dict[class_id], test_text_state[ti])
                    if sim is None or cossim > sim:
                        pc = class_id
                        sim = cossim
                pred_mat[ti][pc - 1] = 1

            topk = self.get_pred_class_topk(pred_mat, k=1)
            topk_list.append(topk)
            pred_class_list.append(pred_mat)
            # kg_vector_list.append(kg_vector_mat)

            # align_list.append(align_mat)

            # print(align_mat)
            # print(pred_mat)
            # print(pred_class_list)
            # print([np.argmax(pred) + 1 for pred in pred_mat])
            # print(true_class_id_mini)
            # exit()

            if cstep % config.cstep_print_unseen == 0:
                tmp_pred = np.concatenate(pred_class_list, axis=0)
                threshold = 0.5
                tmp_pred[tmp_pred > threshold] = 1
                tmp_pred[tmp_pred <= threshold] = 0
                tmp_gt = self.get_one_hot_results(
                    np.reshape(np.array(class_list[: (cstep + 1) * config.batch_size]), newshape=(-1, 1)))
                tmp_stats = utils.get_statistics(tmp_pred, tmp_gt, single_label_pred=False)

                print(
                    "[Test] Epoch: [%3d][%4d/%4d] time: %.4f, \n %s" %
                    (epoch, cstep, test_steps, time.time() - step_time, utils.dict_to_string_4_print(tmp_stats))
                )
                step_time = time.time()

        prediction = np.concatenate(pred_class_list, axis=0)

        tmp_pred = np.concatenate(pred_class_list, axis=0)
        threshold = 0.5
        tmp_pred[tmp_pred > threshold] = 1
        tmp_pred[tmp_pred <= threshold] = 0

        # topk_pred = self.get_one_hot_results(np.concatenate(topk_list, axis=0))
        ground_truth = self.get_one_hot_results(
            np.reshape(np.array(class_list[: test_steps * config.batch_size]), newshape=(-1, 1)))

        # align = np.concatenate(align_list, axis=0)
        # kg_vector_full = np.concatenate(kg_vector_list, axis=0)

        stats = utils.get_statistics(tmp_pred, ground_truth, single_label_pred=False)

        print(
            "[Test Sum] Epoch: [%3d] time: %.4f, \n %s" %
            (epoch, time.time() - start_time, utils.dict_to_string_4_print(stats))
        )

        # return stats, prediction, ground_truth, align, kg_vector_full
        return stats, prediction, ground_truth, np.array([0]), np.array([0])

    def __test__(self, epoch, text_seqs, class_list, unseen_only=False):
        assert config.model != "autoencoder"
        assert len(text_seqs) == len(class_list)

        start_time = time.time()
        step_time = time.time()

        print("the total test sequence is ",len(text_seqs))
        test_steps = math.ceil(len(text_seqs) / config.batch_size)
        print("the total test_steps is ", test_steps, config.batch_size)

        topk_list = list()
        pred_class_list = list()
        # align_list = list()

        # kg_vector_list = list()

        for cstep in range(test_steps):
            text_seqs_mini = text_seqs[cstep * config.batch_size: (cstep + 1) * config.batch_size]
            true_class_id_mini = class_list[cstep * config.batch_size: (cstep + 1) * config.batch_size]

            if len(text_seqs_mini) < config.batch_size:
                print("inputting dummies")
                dummy_sentence = [[1]] * (config.batch_size - len(text_seqs_mini))
                dummy_true_class_id = [1] * (config.batch_size - len(text_seqs_mini))
                text_seqs_mini = text_seqs_mini + dummy_sentence
                true_class_id_mini = true_class_id_mini + dummy_true_class_id
                print(text_seqs_mini, true_class_id_mini)
                print("finished inputting dummies")

            encode_seqs_id_mini, encode_seqs_mat_mini = self.prepro_encode(text_seqs_mini)

            pred_mat = np.zeros([config.batch_size, len(self.class_dict)])
            # align_mat = np.zeros([config.batch_size, len(self.class_dict), self.model.max_length - 1])
            # kg_vector_mat = np.zeros([config.batch_size, len(self.class_dict), config.max_length, config.kg_embedding_dim])

            for class_id in self.class_dict:

                if unseen_only and class_id not in self.unseen_class:
                    continue

                class_id_mini = np.array([class_id] * config.batch_size)
                class_label_embed_mini = self.get_class_label_embed(class_id_mini)
                category_logits = np.zeros([config.batch_size, 1])
                for b in range(config.batch_size):
                    category_logits[b, 0] = int(class_id == true_class_id_mini[b])
                if config.model == "cnnfc" or config.model == "rnnfc":
                    test_loss, pred = self.sess.run([
                        self.model.test_loss,
                        self.model.test_net.outputs,
                    ], feed_dict={
                        self.model.encode_seqs: encode_seqs_mat_mini,
                        self.model.class_label_seqs: class_label_embed_mini,
                        self.model.category_logits: category_logits,
                    })
                else:
                    if config.model == "vwvc":
                        kg_vector_seqs_mini = np.zeros(
                            (config.batch_size, self.model.max_length, self.model.kg_embedding_dim))
                    else:
                        kg_vector_seqs_mini = self.get_kg_vector_given_class(encode_seqs_id_mini, class_id_mini)
                    test_loss, pred = self.sess.run([
                        self.model.test_loss,
                        self.model.test_net.outputs,
                        # self.model.test_align.outputs,
                    ], feed_dict={
                        self.model.encode_seqs: encode_seqs_mat_mini,
                        self.model.class_label_seqs: class_label_embed_mini,
                        self.model.kg_vector: kg_vector_seqs_mini,
                        self.model.category_logits: category_logits,
                    })

                # print(test_loss)

                pred_mat[:, class_id - 1] = pred[:, 0]
                # align_mat[:, class_id - 1, :] = align[:, :, 0]
                # kg_vector_mat[:, class_id - 1, :, :] = kg_vector_seqs_mini

            topk = self.get_pred_class_topk(pred_mat, k=1)
            topk_list.append(topk)
            pred_class_list.append(pred_mat)
            # kg_vector_list.append(kg_vector_mat)

            # align_list.append(align_mat)

            # print(align_mat)
            # print(pred_mat)
            # print(pred_class_list)
            # print([np.argmax(pred) + 1 for pred in pred_mat])
            # print(true_class_id_mini)
            # exit()

            if cstep % config.cstep_print_unseen == 0:
                tmp_pred = np.concatenate(pred_class_list, axis=0)
                threshold = 0.5
                tmp_pred[tmp_pred > threshold] = 1
                tmp_pred[tmp_pred <= threshold] = 0
                tmp_gt = self.get_one_hot_results(
                    np.reshape(np.array(class_list[: (cstep + 1) * config.batch_size]), newshape=(-1, 1)))
                tmp_stats = utils.get_statistics(tmp_pred, tmp_gt, single_label_pred=False)

                print(
                    "[Test] Epoch: [%3d][%4d/%4d] time: %.4f, \n %s" %
                    (epoch, cstep, test_steps, time.time() - step_time, utils.dict_to_string_4_print(tmp_stats))
                )
                step_time = time.time()

        prediction = np.concatenate(pred_class_list, axis=0)
        tmp_pred = np.concatenate(pred_class_list, axis=0)

        prediction = prediction[:len(text_seqs)]
        tmp_pred = tmp_pred[:len(text_seqs)]

        threshold = 0.5
        tmp_pred[tmp_pred > threshold] = 1
        tmp_pred[tmp_pred <= threshold] = 0

        # topk_pred = self.get_one_hot_results(np.concatenate(topk_list, axis=0))
        ground_truth = self.get_one_hot_results(
            np.reshape(np.array(class_list[: test_steps * config.batch_size]), newshape=(-1, 1)))

        # align = np.concatenate(align_list, axis=0)
        # kg_vector_full = np.concatenate(kg_vector_list, axis=0)

        stats = utils.get_statistics(tmp_pred, ground_truth, single_label_pred=False)

        print(
            "[Test Sum] Epoch: [%3d] time: %.4f, \n %s" %
            (epoch, time.time() - start_time, utils.dict_to_string_4_print(stats))
        )

        # return stats, prediction, ground_truth, align, kg_vector_full
        return stats, prediction, ground_truth, np.array([0]), np.array([0])

    def controller(self, train_text_seqs, train_class_list, train_aug_text_seqs, train_aug_class_list, test_text_seqs,
                   test_class_list, rgroup, train_epoch=config.train_epoch, save_test_per_epoch=1):

        last_save_epoch = self.base_epoch
        global_epoch = self.base_epoch + 1

        if last_save_epoch >= 0:
            self.restore_model(
                path=self.model_save_dir,
                global_step=last_save_epoch
            )

        train_text_seqs, train_class_list = self.get_text_of_seen_class(train_text_seqs, train_class_list)
        train_aug_text_seqs, train_aug_class_list = self.get_text_of_unseen_class(train_aug_text_seqs,
                                                                                  train_aug_class_list, augdata=True)

        assert len(train_aug_class_list) == len(train_aug_text_seqs)
        assert len(train_aug_class_list) <= config.augmentation * len(self.unseen_class)
        print("Augmented data from this random group keep unseen only: ", utils.counter_of_list(train_aug_class_list))
        print("Origin real data from seen classes:", utils.counter_of_list(train_class_list))

        seen_test_text_seqs, seen_test_class_list = self.get_text_of_seen_class(test_text_seqs, test_class_list)
        unseen_test_text_seqs, unseen_test_class_list = self.get_text_of_unseen_class(test_text_seqs, test_class_list)

        # double check if text of unseen classes are removed
        for class_id in train_class_list:
            assert self.check_seen(class_id)
        # seen
        for class_id in seen_test_class_list:
            assert self.check_seen(class_id)
        # unseen
        for class_id in unseen_test_class_list:
            assert not self.check_seen(class_id)

        for epoch in range(train_epoch + 1):

            # TODO: apply constrain on training set size
            # for _ in range(max(10 - 10 * global_epoch, 10)):
            # config.small_epoch = 5 # 20news
            # config.small_epoch = 1 # dbpedia
            for _ in range(config.small_epoch):
                print("epoch %d %d/%d" % (global_epoch, _ + 1, config.small_epoch))
                if config.augmentation > 0:
                    print("Training with augmentation %s" % (global_epoch > 6))
                    self.__train__(
                        global_epoch,
                        train_text_seqs + train_aug_text_seqs if global_epoch > 6 else train_text_seqs,
                        train_class_list + train_aug_class_list if global_epoch > 6 else train_class_list,
                        max_train_steps=300
                    )
                else:
                    print("Training without augmentation")

                    self.__train__(
                        global_epoch,
                        train_text_seqs,
                        train_class_list,
                        max_train_steps=1000 \
                            if not (config.dataset == "dbpedia" and config.unseen_rate == 0.5 and "kg" in config.model) \
                            else 300
                    )

            if global_epoch > self.base_epoch and global_epoch % save_test_per_epoch == 0:
                self.save_model(
                    path=self.model_save_dir,
                    global_step=global_epoch
                )
                last_save_epoch = global_epoch
            """
            if global_epoch % save_test_per_epoch == 0:
                
                if config.model == "autoencoder":
                    # TODO: remove to get full test
                    print("[Test] Testing seen classes")
                    stats_seen, pred_seen, gt_seen, align_seen, kg_vector_seen = self.__test_autoencoder__(
                        global_epoch,
                        [_ for idx, _ in enumerate(seen_test_text_seqs) if idx % 50 == 0],
                        [_ for idx, _ in enumerate(seen_test_class_list) if idx % 50 == 0],
                    )

                    # TODO: remove to get full test
                    test_percentage = 10 if config.unseen_rate == 0.25 else 20
                    print("[Test] Testing unseen classes")
                    stats_unseen, pred_unseen, gt_unseen, align_unseen, kg_vector_unseen = self.__test_autoencoder__(
                        global_epoch,
                        [_ for idx, _ in enumerate(unseen_test_text_seqs) if idx % test_percentage == 0],
                        [_ for idx, _ in enumerate(unseen_test_class_list) if idx % test_percentage == 0],
                    )
                else:
                    # TODO: remove to get full test
                    #print("[Test] Testing seen classes")
                    #stats_seen, pred_seen, gt_seen, align_seen, kg_vector_seen = self.__test__(
                    #    global_epoch,
                    #    [_ for idx, _ in enumerate(seen_test_text_seqs) if idx % 50 == 0],
                    #    [_ for idx, _ in enumerate(seen_test_class_list) if idx % 50 == 0],
                    #)

                    # TODO: remove to get full test
                    test_percentage = 10 if config.unseen_rate == 0.25 else 20
                    print("[Test] Testing unseen classes")
                    stats_unseen, pred_unseen, gt_unseen, align_unseen, kg_vector_unseen = self.__test__(
                        global_epoch,
                        [_ for idx, _ in enumerate(unseen_test_text_seqs) if idx % test_percentage == 0],
                        [_ for idx, _ in enumerate(unseen_test_class_list) if idx % test_percentage == 0],
                        unseen_only=False if config.model == "cnnfc" or config.model == "rnnfc" else True
                    )

                np.savez(
                    self.log_save_dir + "test_%d" % global_epoch,
                    seen_class=self.seen_class,
                    unseen_class=self.unseen_class,
                    pred_seen=pred_seen,
                    pred_unseen=pred_unseen,
                    gt_seen=gt_seen,
                    gt_unseen=gt_unseen,
                    align_seen=align_seen,
                    align_unseen=align_unseen,
                    kg_vector_seen=kg_vector_seen,
                    kg_vector_unseen=kg_vector_unseen,
                )
                # error.rejection_single_label(self.log_save_dir + "test_%d.npz" % global_epoch)
                # error.classify_single_label(self.log_save_dir + "test_%d.npz" % global_epoch)
                error.classify_single_label2(self.log_save_dir + "test_%d.npz" % global_epoch)
                # error.classify_single_label_for_seen(self.log_save_dir + "test_%d.npz" % global_epoch, rgroup)
                # error.classify_single_label_for_unseen(self.log_save_dir + "test_%d.npz" % global_epoch, rgroup)
                # class_distance_matrix = np.loadtxt(config.zhang15_dbpedia_dir + 'class_distance.txt')
                # error.classify_adjust_single_label(self.log_save_dir + "test_%d.npz" % global_epoch, class_distance_matrix)
                """
            global_epoch += 1

    def controller4test(self, test_text_seqs, test_class_list, unseen_class_list, rgroup, base_epoch=None):
        self.unseen_class = unseen_class_list

        last_save_epoch = self.base_epoch if base_epoch is None else base_epoch
        global_epoch = self.base_epoch if base_epoch is None else base_epoch

        if last_save_epoch >= 0:
            self.restore_model(
                path=self.model_save_dir,
                global_step=last_save_epoch
            )
        print('shape of test_text_seqs', len(test_text_seqs))
        seen_test_text_seqs, seen_test_class_list = self.get_text_of_seen_class(test_text_seqs, test_class_list)
        unseen_test_text_seqs, unseen_test_class_list = self.get_text_of_unseen_class(test_text_seqs, test_class_list)

        # seen
        for class_id in seen_test_class_list:
            assert self.check_seen(class_id)
        # unseen
        for class_id in unseen_test_class_list:
            assert not self.check_seen(class_id)

        if config.model == "autoencoder":
            print("[Test] Testing seen classes")
            state_seen, pred_seen, gt_seen, align_seen, kg_vector_seen = self.__test_autoencoder__(
                global_epoch,
                [_ for idx, _ in enumerate(seen_test_text_seqs) if idx % 1 == 0],
                [_ for idx, _ in enumerate(seen_test_class_list) if idx % 1 == 0],
            )
            print("[Test] Testing unseen classes")
            state_unseen, pred_unseen, gt_unseen, align_unseen, kg_vector_unseen = self.__test_autoencoder__(
                global_epoch,
                [_ for idx, _ in enumerate(unseen_test_text_seqs) if idx % 1 == 0],
                [_ for idx, _ in enumerate(unseen_test_class_list) if idx % 1 == 0],
            )
        else:
            # TODO: remove to get full test
            if config.model == "cnnfc" or config.model == "rnnfc":
                print("[Test] Testing seen classes")
                state_seen, pred_seen, gt_seen, align_seen, kg_vector_seen = self.__test__(
                    global_epoch,
                    [_ for idx, _ in enumerate(seen_test_text_seqs) if idx % 1 == 0],
                    [_ for idx, _ in enumerate(seen_test_class_list) if idx % 1 == 0],
                )
            else:
                state_seen = pred_seen = gt_seen = align_seen = kg_vector_seen = 0

            print("length of unseen sequence", len(unseen_test_text_seqs))

            # TODO: remove to get full test
            print("[Test] Testing unseen classes for", config.model)
            state_unseen, pred_unseen, gt_unseen, align_unseen, kg_vector_unseen = self.__test__(
                global_epoch,
                [_ for idx, _ in enumerate(unseen_test_text_seqs) if idx % 1 == 0],
                [_ for idx, _ in enumerate(unseen_test_class_list) if idx % 1 == 0],
                unseen_only=False if config.model == "cnnfc" or config.model == "rnnfc" else True
            )

            print('The prediction is......', pred_unseen.shape, state_unseen)
            print(pred_unseen)
            original_text_dataframe = dataloader.load_text_direct(config.imdb_test_path)
            original_text_dataframe = pd.concat([original_text_dataframe, pd.DataFrame(pred_unseen)], axis=1)
            print(original_text_dataframe)
            original_text_dataframe.to_excel(config.imdb_prediction_result_path, engine='xlsxwriter')

        np.savez(
            self.log_save_dir + "test_full_%d" % global_epoch,
            seen_class=self.seen_class,
            unseen_class=self.unseen_class,
            pred_seen=pred_seen,
            pred_unseen=pred_unseen,
            gt_seen=gt_seen,
            gt_unseen=gt_unseen,
            align_seen=align_seen,
            align_unseen=align_unseen,
            kg_vector_seen=kg_vector_seen,
            kg_vector_unseen=kg_vector_unseen,
        )

        # error.rejection_single_label(self.log_save_dir + "test_%d.npz" % global_epoch)
        # error.classify_single_label(self.log_save_dir + "test_%d.npz" % global_epoch)
        # error.classify_single_label2(self.log_save_dir + "test_full_%d.npz" % global_epoch)
        error.classify_single_label_for_unseen(self.log_save_dir + "test_full_%d.npz" % global_epoch, rgroup)
        # class_distance_matrix = np.loadtxt(config.zhang15_dbpedia_dir + 'class_distance.txt')
        # error.classify_adjust_single_label(self.log_save_dir + "test_%d.npz" % global_epoch, class_distance_matrix)


def run_dbpedia():
    random_group = dataloader.get_random_group(config.zhang15_dbpedia_class_random_group_path)

    vocab = dataloader.build_vocabulary_from_full_corpus(
        # config.zhang15_dbpedia_full_data_path, config.zhang15_dbpedia_vocab_path, column="selected", force_process=False,
        config.zhang15_dbpedia_full_data_path, config.zhang15_dbpedia_vocab_path, column="text", force_process=False,
        min_word_count=55
    )

    glove_mat = dataloader.load_glove_word_vector(
        config.word_embed_file_path, config.zhang15_dbpedia_word_embed_matrix_path, vocab, force_process=False
    )
    assert np.sum(glove_mat[vocab.start_id]) == 0
    assert np.sum(glove_mat[vocab.end_id]) == 0
    assert np.sum(glove_mat[vocab.unk_id]) == 0
    assert np.sum(glove_mat[vocab.pad_id]) == 0

    class_dict = dataloader.load_class_dict(
        class_file=config.zhang15_dbpedia_class_label_path,
        class_code_column="ClassCode",
        class_name_column="ConceptNet"
    )

    # check class label in vocab and glove
    for class_id in class_dict:
        class_label = class_dict[class_id]
        class_label_word_id = vocab.word_to_id(class_label)
        assert class_label_word_id != vocab.unk_id
        assert np.sum(glove_mat[class_label_word_id]) != 0

    if config.model == "cnnfc" or config.model == "rnnfc" or config.model == "vwvc" or config.model == "autoencoder":
        kg_vector_dict = dict()
    else:
        kg_vector_dict = dataloader.load_kg_vector(
            config.zhang15_dbpedia_kg_vector_dir,
            config.zhang15_dbpedia_kg_vector_prefix,
            class_dict
        )

    print("Check NaN in csv ...")
    check_nan_train = dataloader.check_df(config.zhang15_dbpedia_train_path)
    check_nan_test = dataloader.check_df(config.zhang15_dbpedia_test_path)
    print("Train NaN %s, Test NaN %s" % (check_nan_train, check_nan_test))

    assert not check_nan_train
    assert not check_nan_test

    train_class_list = dataloader.load_data_class(
        filename=config.zhang15_dbpedia_train_path,
        column="class",
    )

    train_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.zhang15_dbpedia_train_path, vocab, config.zhang15_dbpedia_train_processed_path,
        column="text", force_process=False
        # column="selected", force_process=False
        # column="selected_tfidf", force_process=False
    )

    test_class_list = dataloader.load_data_class(
        filename=config.zhang15_dbpedia_test_path,
        column="class",
    )

    test_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.zhang15_dbpedia_test_path, vocab, config.zhang15_dbpedia_test_processed_path,
        column="text", force_process=False
        # column="selected", force_process=False
        # column="selected_tfidf", force_process=False
    )

    for class_id in test_class_list + train_class_list:
        assert class_id in class_dict

    if config.augmentation > 0:
        train_aug_from_class_list = dataloader.load_data_class(
            filename=config.zhang15_dbpedia_train_augmented_path,
            column="from_class",
        )

        train_aug_class_list = dataloader.load_data_class(
            filename=config.zhang15_dbpedia_train_augmented_path,
            column="to_class",
        )

        train_aug_text_seqs = dataloader.load_data_from_text_given_vocab(
            config.zhang15_dbpedia_train_augmented_path, vocab, config.zhang15_dbpedia_train_augmented_processed_path,
            column="text", force_process=False
        )

        print("Augmented data size origin: ", utils.counter_of_list(train_aug_class_list))

    else:
        train_aug_from_class_list = []
        train_aug_class_list = []
        train_aug_text_seqs = []

    assert len(train_aug_class_list) == len(train_aug_text_seqs)

    lenlist = [len(text) for text in test_text_seqs] + [len(text) for text in train_text_seqs]
    print("Avg length of documents: ", np.mean(lenlist))
    print("95% length of documents: ", np.percentile(lenlist, 95))
    print("90% length of documents: ", np.percentile(lenlist, 90))
    print("80% length of documents: ", np.percentile(lenlist, 80))
    # exit()

    for i, rgroup in enumerate(random_group):

        if i + 1 < config.random_group_start_idx:
            continue

        max_length = 80

        print("Removing augmented data from unseen classes ...")
        train_aug_class_list_from_seen = list()
        train_aug_text_seqs_from_seen = list()
        for tidx, text in enumerate(train_aug_text_seqs):
            if train_aug_from_class_list[tidx] in rgroup[1]:
                continue
            train_aug_class_list_from_seen.append(train_aug_class_list[tidx])
            train_aug_text_seqs_from_seen.append(train_aug_text_seqs[tidx])
        assert len(train_aug_class_list_from_seen) == len(train_aug_text_seqs_from_seen)
        print("Augmented data from seen class: %d" % len(train_aug_class_list_from_seen),
              utils.counter_of_list(train_aug_class_list_from_seen))

        with tf.Graph().as_default() as graph:
            tl.layers.clear_layers_name()

            mdl = model_unseen.Model4Unseen(
                # model_name="unseen_full_zhang15_dbpedia_kg3_cluster_3group_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext_aug%d" \
                # model_name="unseen_full_zhang15_dbpedia_kg3_cluster_3group_kgonly_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext_aug%d" \
                model_name="unseen_full_zhang15_dbpedia_kg3_cluster_3group_%s_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext_aug%d" \
                           % (
                               config.model, i + 1, "-".join(str(_) for _ in rgroup[1]), max_length,
                               config.negative_sample,
                               config.negative_increase, config.augmentation),
                start_learning_rate=0.0001,
                decay_rate=0.8,
                decay_steps=2e3,
                max_length=max_length,
                vocab_size=vocab.unk_id + 1,
            )
            # TODO: if unseen_classes are already selected, set randon_unseen_class=False and provide a list of unseen_classes
            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.per_process_gpu_memory_fraction = config.global_gpu_occupation
            ctl = Controller4Unseen(
                model=mdl,
                vocab=vocab,
                gpu_config=gpu_config,
                class_dict=class_dict,
                kg_vector_dict=kg_vector_dict,
                word_embed_mat=glove_mat,
                lemma=True,
                random_unseen_class=False,
                random_unseen_class_list=rgroup[1],
                base_epoch=-1,
            )
            if config.global_is_train:
                ctl.controller(train_text_seqs, train_class_list,
                               train_aug_text_seqs_from_seen, train_aug_class_list_from_seen,
                               test_text_seqs, test_class_list,
                               rgroup=rgroup, train_epoch=10,
                               save_test_per_epoch=5 if config.model in ["vwvkg", "vcvkg", "kgonly"] else 1)
            else:
                ctl.controller4test(test_text_seqs, test_class_list, unseen_class_list=ctl.unseen_class,
                                    rgroup=rgroup, base_epoch=config.global_test_base_epoch)

            ctl.sess.close()
    pass


def run_20news():
    random_group = dataloader.get_random_group(config.news20_class_random_group_path)

    vocab = dataloader.build_vocabulary_from_full_corpus(
        config.news20_full_data_path, config.news20_vocab_path, column="text", force_process=False,
        min_word_count=10
    )

    glove_mat = dataloader.load_glove_word_vector(
        config.word_embed_file_path, config.news20_word_embed_matrix_path, vocab, force_process=False
    )
    assert np.sum(glove_mat[vocab.start_id]) == 0
    assert np.sum(glove_mat[vocab.end_id]) == 0
    assert np.sum(glove_mat[vocab.unk_id]) == 0
    assert np.sum(glove_mat[vocab.pad_id]) == 0

    class_dict = dataloader.load_class_dict(
        class_file=config.news20_class_label_path,
        class_code_column="ClassCode",
        class_name_column="ConceptNet"
    )

    # check class label in vocab and glove
    for class_id in class_dict:
        class_label = class_dict[class_id]
        class_label_word_id = vocab.word_to_id(class_label)
        assert class_label_word_id != vocab.unk_id
        assert np.sum(glove_mat[class_label_word_id]) != 0

    if config.model == "cnnfc" or config.model == "rnnfc" or config.model == "vwvc" or config.model == "autoencoder":
        kg_vector_dict = dict()
    else:
        kg_vector_dict = dataloader.load_kg_vector(
            config.news20_kg_vector_dir,
            config.news20_kg_vector_prefix,
            class_dict
        )

    print("Check NaN in csv ...")
    check_nan_train = dataloader.check_df(config.news20_train_path)
    check_nan_test = dataloader.check_df(config.news20_test_path)
    print("Train NaN %s, Test NaN %s" % (check_nan_train, check_nan_test))
    assert not check_nan_train
    assert not check_nan_test

    train_class_list = dataloader.load_data_class(
        filename=config.news20_train_path,
        column="class",
    )

    train_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.news20_train_path, vocab, config.news20_train_processed_path,
        # column="text", force_process=False
        # column="selected", force_process=False
        column="selected_tfidf", force_process=False
    )

    test_class_list = dataloader.load_data_class(
        filename=config.news20_test_path,
        column="class",
    )

    test_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.news20_test_path, vocab, config.news20_test_processed_path,
        # column="text", force_process=False
        # column="selected", force_process=False
        column="selected_tfidf", force_process=False
    )

    for class_id in test_class_list + train_class_list:
        assert class_id in class_dict

    if config.augmentation > 0:
        train_aug_from_class_list = dataloader.load_data_class(
            filename=config.news20_train_augmented_path,
            column="from_class",
        )

        train_aug_class_list = dataloader.load_data_class(
            filename=config.news20_train_augmented_path,
            column="to_class",
        )

        train_aug_text_seqs = dataloader.load_data_from_text_given_vocab(
            config.news20_train_augmented_path, vocab, config.news20_train_augmented_processed_path,
            column="text", force_process=False
        )

        print("Augmented data size origin: ", utils.counter_of_list(train_aug_class_list))

    else:
        train_aug_from_class_list = []
        train_aug_class_list = []
        train_aug_text_seqs = []

    print(len(train_aug_class_list), len(train_aug_text_seqs))
    assert len(train_aug_class_list) == len(train_aug_text_seqs)

    lenlist = [len(text) for text in test_text_seqs] + [len(text) for text in train_text_seqs]
    print("Avg length of documents: ", np.mean(lenlist))
    print("95% length of documents: ", np.percentile(lenlist, 95))
    print("90% length of documents: ", np.percentile(lenlist, 90))
    print("80% length of documents: ", np.percentile(lenlist, 80))
    # exit()

    for i, rgroup in enumerate(random_group):

        if i + 1 < config.random_group_start_idx:
            continue

        max_length = 50

        print("Removing augmented data from unseen classes ...")
        train_aug_class_list_from_seen = list()
        train_aug_text_seqs_from_seen = list()
        for tidx, text in enumerate(train_aug_text_seqs):
            if train_aug_from_class_list[tidx] in rgroup[1]:
                continue
            train_aug_class_list_from_seen.append(train_aug_class_list[tidx])
            train_aug_text_seqs_from_seen.append(train_aug_text_seqs[tidx])
        assert len(train_aug_class_list_from_seen) == len(train_aug_text_seqs_from_seen)
        print("Augmented data from seen class: %d" % len(train_aug_class_list_from_seen),
              utils.counter_of_list(train_aug_class_list_from_seen))

        with tf.Graph().as_default() as graph:
            tl.layers.clear_layers_name()

            mdl = model_unseen.Model4Unseen(
                # model_name="unseen_selected_tfidf_news20_kg3_cluster_3group_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext" \
                # model_name="unseen_selected_tfidf_news20_kg3_cluster_3group_only_smallepoch5_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext" \
                # model_name="unseen_selected_tfidf_news20_kg3_cluster_3group_kgonly_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext_aug%d" \
                model_name="unseen_selected_tfidf_news20_kg3_cluster_3group_%s_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext_aug%d" \
                           % (
                               config.model, i + 1, "-".join(str(_) for _ in rgroup[1]), max_length,
                               config.negative_sample,
                               config.negative_increase, config.augmentation),
                start_learning_rate=0.0001,
                decay_rate=0.5,
                decay_steps=600,
                max_length=max_length,
                vocab_size=vocab.unk_id + 1,
            )
            # TODO: if unseen_classes are already selected, set randon_unseen_class=False and provide a list of unseen_classes
            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.per_process_gpu_memory_fraction = config.global_gpu_occupation
            ctl = Controller4Unseen(
                model=mdl,
                vocab=vocab,
                gpu_config=gpu_config,
                class_dict=class_dict,
                kg_vector_dict=kg_vector_dict,
                word_embed_mat=glove_mat,
                lemma=True,
                random_unseen_class=False,
                random_unseen_class_list=rgroup[1],
                base_epoch=-1,
            )
            if config.global_is_train:
                ctl.controller(train_text_seqs, train_class_list,
                               train_aug_text_seqs_from_seen, train_aug_class_list_from_seen,
                               test_text_seqs, test_class_list, rgroup=rgroup, train_epoch=10)
            else:
                ctl.controller4test(test_text_seqs, test_class_list, unseen_class_list=ctl.unseen_class,
                                    rgroup=rgroup, base_epoch=config.global_test_base_epoch)

            ctl.sess.close()
    pass


def run_amazon():
    random_group = dataloader.get_random_group(config.chen14_elec_class_random_group_path)

    vocab = dataloader.build_vocabulary_from_full_corpus(
        config.chen14_elec_full_data_path, config.chen14_elec_vocab_path, column="text", force_process=False,
        min_word_count=10
    )

    glove_mat = dataloader.load_glove_word_vector(
        config.word_embed_file_path, config.chen14_elec_word_embed_matrix_path, vocab, force_process=False
    )
    assert np.sum(glove_mat[vocab.start_id]) == 0
    assert np.sum(glove_mat[vocab.end_id]) == 0
    assert np.sum(glove_mat[vocab.unk_id]) == 0
    assert np.sum(glove_mat[vocab.pad_id]) == 0

    class_dict = dataloader.load_class_dict(
        class_file=config.chen14_elec_class_label_path,
        class_code_column="ClassCode",
        class_name_column="ConceptNet"
    )

    # check class label in vocab and glove
    for class_id in class_dict:
        class_label = class_dict[class_id]
        class_label_word_id = vocab.word_to_id(class_label)
        assert class_label_word_id != vocab.unk_id
        assert np.sum(glove_mat[class_label_word_id]) != 0

    kg_vector_dict = dataloader.load_kg_vector(
        config.chen14_elec_kg_vector_dir,
        config.chen14_elec_kg_vector_prefix,
        class_dict
    )

    print("Check NaN in csv ...")
    check_nan_train = dataloader.check_df(config.chen14_elec_train_path)
    check_nan_test = dataloader.check_df(config.chen14_elec_test_path)
    print("Train NaN %s, Test NaN %s" % (check_nan_train, check_nan_test))
    assert not check_nan_train
    assert not check_nan_test

    train_class_list = dataloader.load_data_class(
        filename=config.chen14_elec_train_path,
        column="class",
    )

    train_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.chen14_elec_train_path, vocab, config.chen14_elec_train_processed_path,
        column="text", force_process=False
        # column="selected", force_process=False
        # column="selected_tfidf", force_process=False
    )

    test_class_list = dataloader.load_data_class(
        filename=config.chen14_elec_test_path,
        column="class",
    )

    test_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.chen14_elec_test_path, vocab, config.chen14_elec_test_processed_path,
        column="text", force_process=False
        # column="selected", force_process=False
        # column="selected_tfidf", force_process=False
    )

    lenlist = [len(text) for text in test_text_seqs] + [len(text) for text in train_text_seqs]
    print("Avg length of documents: ", np.mean(lenlist))
    print("95% length of documents: ", np.percentile(lenlist, 95))
    print("90% length of documents: ", np.percentile(lenlist, 90))
    print("80% length of documents: ", np.percentile(lenlist, 80))
    # exit()

    for i, rgroup in enumerate(random_group):

        if i != 0:
            continue

        max_length = 100

        with tf.Graph().as_default() as graph:
            tl.layers.clear_layers_name()

            mdl = model_unseen.Model4Unseen(
                model_name="unseen_full_chen14_elec_kg3_cluster_3group_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext" \
                           % (i + 1, "-".join(str(_) for _ in rgroup[1]), max_length, config.negative_sample,
                              config.negative_increase),
                start_learning_rate=0.0001,
                decay_rate=0.5,
                decay_steps=2000,
                max_length=max_length
            )
            # TODO: if unseen_classes are already selected, set randon_unseen_class=False and provide a list of unseen_classes
            ctl = Controller4Unseen(
                model=mdl,
                vocab=vocab,
                class_dict=class_dict,
                kg_vector_dict=kg_vector_dict,
                word_embed_mat=glove_mat,
                lemma=True,
                random_unseen_class=False,
                random_unseen_class_list=rgroup[1],
                base_epoch=-1,
            )
            ctl.controller(train_text_seqs, train_class_list, test_text_seqs, test_class_list, rgroup=rgroup,
                           train_epoch=10)
            # ctl.controller4test(test_text_seqs, test_class_list, unseen_class_list=ctl.unseen_class, rgroup=rgroup, base_epoch=5)

            ctl.sess.close()
    pass


def run_imdb():
    random_group = dataloader.get_random_group(config.imdb_class_random_group_path)

    vocab = dataloader.build_vocabulary_from_full_corpus(
        config.imdb_full_data_path, config.imdb_vocab_path, column="text", force_process=False,
        min_word_count=5
    )

    glove_mat = dataloader.load_glove_word_vector(
        config.word_embed_file_path, config.imdb_word_embed_matrix_path, vocab, force_process=False
    )

    assert np.sum(glove_mat[vocab.start_id]) == 0
    assert np.sum(glove_mat[vocab.end_id]) == 0
    assert np.sum(glove_mat[vocab.unk_id]) == 0
    assert np.sum(glove_mat[vocab.pad_id]) == 0

    class_dict = dataloader.load_class_dict(
        class_file=config.imdb_class_label_path,
        class_code_column="ClassCode",
        class_name_column="ConceptNet"
    )

    # check class label in vocab and glove
    for class_id in class_dict:
        class_label = class_dict[class_id]
        class_label_word_id = vocab.word_to_id(class_label)
        #print(class_label_word_id, vocab.unk_id)
        assert class_label_word_id != vocab.unk_id
        assert np.sum(glove_mat[class_label_word_id]) != 0


    if config.model == "cnnfc" or config.model == "rnnfc" or config.model == "vwvc" or config.model == "autoencoder":
        kg_vector_dict = dict()
    else:
        kg_vector_dict = dataloader.load_kg_vector(
            config.imdb_kg_vector_dir,
            config.imdb_kg_vector_prefix,
            class_dict
        )

    print("Check NaN in csv ...")
    check_nan_train = dataloader.check_df(config.imdb_train_path)
    check_nan_test = dataloader.check_df(config.imdb_test_path)
    print("Train NaN %s, Test NaN %s" % (check_nan_train, check_nan_test))
    assert not check_nan_train
    assert not check_nan_test

    train_class_list = dataloader.load_data_class(
        filename=config.imdb_train_path,
        column="class",
    )

    train_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.imdb_train_path, vocab, config.imdb_train_processed_path,
        column="text", force_process=False
        # column="selected", force_process=False
        #column="selected_tfidf", force_process=False
    )

    test_class_list = dataloader.load_data_class(
        filename=config.imdb_test_path,
        column="class",
    )

    test_text_seqs = dataloader.load_data_from_text_given_vocab(
        config.imdb_test_path, vocab, config.imdb_test_processed_path,
         column="text", force_process=False
        # column="selected", force_process=False
        # column="selected_tfidf", force_process=False
    )

    for class_id in test_class_list + train_class_list:
        assert class_id in class_dict

    if config.augmentation > 0:
        train_aug_from_class_list = dataloader.load_data_class(
            filename=config.imdb_train_augmented_aggregated_path,
            column="from_class",
        )

        train_aug_class_list = dataloader.load_data_class(
            filename=config.imdb_train_augmented_aggregated_path,
            column="to_class",
        )

        train_aug_text_seqs = dataloader.load_data_from_text_given_vocab(
            config.imdb_train_augmented_aggregated_path, vocab, config.imdb_train_augmented_processed_path,
            column="text", force_process=False
        )

        print("Augmented data size origin: ", utils.counter_of_list(train_aug_class_list))

    else:
        train_aug_from_class_list = []
        train_aug_class_list = []
        train_aug_text_seqs = []

    print(len(train_aug_class_list), len(train_aug_text_seqs))
    assert len(train_aug_class_list) == len(train_aug_text_seqs)

    lenlist = [len(text) for text in test_text_seqs] + [len(text) for text in train_text_seqs]
    print("Avg length of documents: ", np.mean(lenlist))
    print("95% length of documents: ", np.percentile(lenlist, 95))
    print("90% length of documents: ", np.percentile(lenlist, 90))
    print("80% length of documents: ", np.percentile(lenlist, 80))
    # exit()

    for i, rgroup in enumerate(random_group):

        if i + 1 < config.random_group_start_idx:
            continue

        max_length = 50

        print("Removing augmented data from unseen classes ...")
        train_aug_class_list_from_seen = list()
        train_aug_text_seqs_from_seen = list()
        for tidx, text in enumerate(train_aug_text_seqs):
            if train_aug_from_class_list[tidx] in rgroup[1]:
                continue
            train_aug_class_list_from_seen.append(train_aug_class_list[tidx])
            train_aug_text_seqs_from_seen.append(train_aug_text_seqs[tidx])
        assert len(train_aug_class_list_from_seen) == len(train_aug_text_seqs_from_seen)
        print("Augmented data from seen class: %d" % len(train_aug_class_list_from_seen),
              utils.counter_of_list(train_aug_class_list_from_seen))

        with tf.Graph().as_default() as graph:
            tl.layers.clear_layers_name()

            mdl = model_unseen.Model4Unseen(
                # model_name="unseen_selected_tfidf_news20_kg3_cluster_3group_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext" \
                # model_name="unseen_selected_tfidf_news20_kg3_cluster_3group_only_smallepoch5_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext" \
                # model_name="unseen_selected_tfidf_news20_kg3_cluster_3group_kgonly_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext_aug%d" \
                model_name="unseen_selected_tfidf_news20_kg3_cluster_3group_%s_random%d_unseen%s_max%d_cnn_negative%dincrease%d_randomtext_aug%d" \
                           % (
                               config.model, i + 1, "-".join(str(_) for _ in rgroup[1]), max_length,
                               config.negative_sample,
                               config.negative_increase, config.augmentation),
                start_learning_rate=0.0001,
                decay_rate=0.5,
                decay_steps=600,
                max_length=max_length,
                vocab_size=vocab.unk_id + 1,
            )
            # TODO: if unseen_classes are already selected, set randon_unseen_class=False and provide a list of unseen_classes
            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.per_process_gpu_memory_fraction = config.global_gpu_occupation
            ctl = Controller4Unseen(
                model=mdl,
                vocab=vocab,
                gpu_config=gpu_config,
                class_dict=class_dict,
                kg_vector_dict=kg_vector_dict,
                word_embed_mat=glove_mat,
                lemma=True,
                random_unseen_class=False,
                random_unseen_class_list=rgroup[1],
                base_epoch=-1,
            )
            if config.global_is_train:
                ctl.controller(train_text_seqs, train_class_list,
                               train_aug_text_seqs_from_seen, train_aug_class_list_from_seen,
                               test_text_seqs, test_class_list, rgroup=rgroup, train_epoch=10)
            else:
                ctl.controller4test(test_text_seqs, test_class_list, unseen_class_list=ctl.unseen_class,
                                    rgroup=rgroup, base_epoch=config.global_test_base_epoch)
            ctl.sess.close()
    pass


if __name__ == "__main__":

    if config.dataset == "dbpedia":
        run_dbpedia()
    elif config.dataset == "20news":
        run_20news()
    elif config.dataset == "imdb":
        run_imdb()
    else:
        raise Exception("config.dataset %s not found" % config.dataset)
    pass
