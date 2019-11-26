import os
import re
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorlayer as tl
import progressbar

import src_reject.config as config

START_ID = '<START_ID>'
END_ID = '<END_ID>'
PAD_ID = '<PAD_ID>'
UNK_ID = '<UNK_ID>'


def get_random_group(filename):
    random_group = list()
    with open(filename, "r") as f:
        for line in f:
            print(line)
            classlist = line.split("|")
            seen_class = [int(_) for _ in classlist[0].split(",")]
            unseen_class = [int(_) for _ in classlist[1].split(",")]
            random_group.append([seen_class, unseen_class])
    print("Random Group: \n%s" % "\n".join([str(rgroup) for rgroup in random_group]))
    return random_group


def check_df(filename):
    if filename.endswith(".csv"):
        df = pd.read_csv(filename, index_col=0)
    if filename.endswith(".xlsx"):
        df = pd.read_excel(filename, index_col=0)
    nan = df.isnull().values.any()
    if nan:
        print('modifying',filename)
        df.dropna(inplace=True)
        df.reset_index(drop=True)
        if filename.endswith(".csv"):
            df.to_csv(filename)
        if filename.endswith(".xlsx"):
            df.to_excel(filename, engine='xlsxwriter')
    return nan


def preprocess(textlist):
    print("Preprocessing ...")
    with progressbar.ProgressBar(max_value=len(textlist)) as bar:
        for idx, text in enumerate(textlist):
            # textlist[idx].replace(",", " ")
            # textlist[idx].replace(".", " ")
            textlist[idx] = re.sub(r'[\W_]+', ' ', str(text))
            textlist[idx] = tl.nlp.process_sentence(textlist[idx], start_word=START_ID, end_word=END_ID)
            # textlist[idx] = textlist[idx].split() # no empty string in the list
            bar.update(idx + 1)

    return textlist


def create_vocab_given_text(textlist, vocab_path, min_word_count=config.prepro_min_word_count):
    # create dictionary
    tl.nlp.create_vocab(textlist, word_counts_output_file=vocab_path, min_word_count=min_word_count)
    vocab = tl.nlp.Vocabulary(vocab_path, start_word=START_ID, end_word=END_ID, unk_word=UNK_ID)
    return vocab


def sentence_word_to_id(textlist, vocab):
    for idx, text in enumerate(textlist):
        textlist[idx] = [vocab.word_to_id(word) for word in text]
    return textlist


# def prepro_encode_kg_vector(kg_vector_list):
#     for idx, kg_vector in enumerate(kg_vector_list):
#         new_kg_vector = np.zeros([config.max_length, config.kg_embedding_dim])
#         new_kg_vector[:kg_vector.shape[0] - 2, :] = kg_vector[1:-1]
#         kg_vector_list[idx] = new_kg_vector
#     return np.array(kg_vector_list)

def get_text_list(df, column):
    if type(column) == str:
        full_text_list = df[column].tolist()
    elif type(column) == list:
        df["text"] = df[column].apply(
            lambda x: ' '.join([item if type(item) == str else ' ' for item in x]), axis=1)
        full_text_list = df["text"].tolist()
    else:
        raise Exception("column should be either a string or a list of string")
    return full_text_list


def load_data(filename, vocab_file, processed_file, column, min_word_count=config.prepro_min_word_count,
              force_process=False):
    print("Loading data ...")

    if not force_process and os.path.exists(processed_file) and os.path.exists(vocab_file):
        print("Processed data found in local files. Loading ...")
        if processed_file.endswith(".pkl"):
            with open(processed_file, 'rb') as f:
                full_text_list = pickle.load(f)
        else:
            with open(processed_file, "r") as f:
                full_text_list = eval(f.read())
        vocab = tl.nlp.Vocabulary(vocab_file, start_word=START_ID, end_word=END_ID, unk_word=UNK_ID)
    else:
        df = pd.read_csv(filename, index_col=0)

        full_text_list = get_text_list(df, column)
        full_text_list = preprocess(full_text_list)
        vocab = create_vocab_given_text(full_text_list, vocab_path=vocab_file, min_word_count=min_word_count)
        full_text_list = sentence_word_to_id(full_text_list, vocab)

        if processed_file.endswith(".pkl"):
            with open(processed_file, "wb") as f:
                pickle.dump(full_text_list, f)
        else:
            with open(processed_file, "w") as f:
                f.write(str(full_text_list))
        print("Processed data saved to %s" % processed_file)

    print("Data loaded: num of seqs %s" % len(full_text_list))
    return full_text_list, vocab


def build_vocabulary_from_full_corpus(filename, vocab_file, column, min_word_count=config.prepro_min_word_count,
                                      force_process=False):
    if not force_process and os.path.exists(vocab_file):
        print("Load vocab from local file")
        vocab = tl.nlp.Vocabulary(vocab_file, start_word=START_ID, end_word=END_ID, unk_word=UNK_ID)
    else:
        print("Creating vocab ...")
        df = pd.read_excel(filename, index_col=0)
        print(df)

        full_text_list = get_text_list(df, column)
        full_text_list = preprocess(full_text_list)
        vocab = create_vocab_given_text(full_text_list, vocab_path=vocab_file, min_word_count=min_word_count)
        print("Vocab created and saved in %s" % vocab_file)
    return vocab


def load_data_class(filename, column):
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(filename, index_col=0)
        if filename.endswith(".xlsx"):
            df = pd.read_excel(filename, index_col=0)
        df = pd.read_csv(filename, index_col=0)
    except:
        if filename.endswith(".csv"):
            df = pd.read_csv(filename, index_col=0, encoding="utf-8")
        if filename.endswith(".xlsx"):
            df = pd.read_excel(filename, index_col=0)
    data_class_list = df[column].tolist()
    return data_class_list


def load_class_dict(class_file, class_code_column, class_name_column):
    class_df = pd.read_csv(class_file)
    class_dict = dict(zip(class_df[class_code_column], class_df[class_name_column]))

    return class_dict

def load_text_direct(filename):
    print("Loading text ...")
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(filename, index_col=0)
        if filename.endswith(".xlsx"):
            df = pd.read_excel(filename, index_col=0)
        df = pd.read_csv(filename, index_col=0)
    except:
        if filename.endswith(".csv"):
            df = pd.read_csv(filename, index_col=0, encoding="utf-8")
        if filename.endswith(".xlsx"):
            df = pd.read_excel(filename, index_col=0)
    return df

def load_data_from_text_given_vocab(filename, vocab, processed_file, column, force_process=False):
    print("Loading data given vocab ...")

    if not force_process and os.path.exists(processed_file):
        print("Processed data found in local files. Loading ...")
        if processed_file.endswith(".pkl"):
            with open(processed_file, 'rb') as f:
                full_text_list = pickle.load(f)
        else:
            with open(processed_file, "r") as f:
                full_text_list = eval(f.read())
    else:
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(filename, index_col=0)
            if filename.endswith(".xlsx"):
                df = pd.read_excel(filename, index_col=0)
            df = pd.read_csv(filename, index_col=0)
        except:
            if filename.endswith(".csv"):
                df = pd.read_csv(filename, index_col=0, encoding="utf-8")
            if filename.endswith(".xlsx"):
                df = pd.read_excel(filename, index_col=0)
        full_text_list = get_text_list(df, column)
        full_text_list = preprocess(full_text_list)
        full_text_list = sentence_word_to_id(full_text_list, vocab)
        if processed_file.endswith(".pkl"):
            with open(processed_file, "wb") as f:
                pickle.dump(full_text_list, f)
        else:
            with open(processed_file, "w") as f:
                f.write(str(full_text_list))
        print("Processed data saved to %s" % processed_file)
    print("Data loaded: num of seqs %s" % len(full_text_list))
    return full_text_list


def get_kg_vector(kg_vector_dict, class_label, word):
    prefix = '/c/en/'

    if not class_label.startswith(prefix):
        class_label = prefix + class_label.lower()

    assert class_label in kg_vector_dict

    if word in kg_vector_dict[class_label]:
        return kg_vector_dict[class_label][word]
    else:
        if not word.startswith(prefix):
            word = prefix + word.lower()
        if word in kg_vector_dict[class_label]:
            return kg_vector_dict[class_label][word]
        return np.zeros(config.kg_embedding_dim)


def load_kg_vector(filedir, fileprefix, class_dict):
    print("Loading KG_VECTOR ...")
    kg_vector_dict = dict()
    for class_id in class_dict:

        with open("%s%s%s.pickle" % (filedir, fileprefix, class_dict[class_id]), 'rb') as f:
            class_kg_dict = pickle.load(f)

            prefix = "/c/en/"
            class_name = class_dict[class_id]
            if not class_name.startswith(prefix):
                class_name = prefix + class_name

            assert class_name not in kg_vector_dict
            kg_vector_dict[class_name] = class_kg_dict

    print(kg_vector_dict.keys())
    return kg_vector_dict


def load_kg_vector_given_text_seqs(text_seqs, vocab, class_dict, kg_vector_dict, processed_file, force_process=False):
    print("Loading KG Vector ...")
    if not force_process and os.path.exists(processed_file):
        print("Processed data found in local files. Loading ...")
        with open(processed_file, 'rb') as f:
            kg_vector_seqs = pickle.load(f)
    else:
        kg_vector_seqs = list()

        with progressbar.ProgressBar(max_value=len(text_seqs)) as bar:
            for idx, text in enumerate(text_seqs):
                kg_vector_text_dict = dict()
                for class_id in class_dict:
                    kg_vector = np.zeros([len(text), config.kg_embedding_dim])
                    for widx, word_id in enumerate(text):
                        kg_vector[widx, :] = get_kg_vector(kg_vector_dict, class_dict[class_id],
                                                           vocab.id_to_word(word_id))
                    kg_vector_text_dict[class_id] = kg_vector
                kg_vector_seqs.append(kg_vector_text_dict)
                bar.update(idx)
        with open(processed_file, "wb") as f:
            pickle.dump(kg_vector_seqs, f)
    return kg_vector_seqs


def load_glove_word_vector(filename, npzfilename, vocab, force_process=False):
    print("Glove loading ... ")

    if not force_process and os.path.exists(npzfilename):
        print("Glove found in local file")
        glove_mat = np.load(npzfilename)["matrix"]
        print("Glove loaded: mat %s, vocab size %d" % (glove_mat.shape, np.count_nonzero(np.sum(glove_mat, axis=1))))

    else:
        glove_mat = np.zeros((vocab.unk_id + 1, config.word_embedding_dim))

        num = 0
        with progressbar.ProgressBar(max_value=400000) as bar:
            with open(filename, 'r', encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    content = line.replace("\n", "").split(" ")

                    word = content[0]
                    vect = np.array(content[1:]).astype(np.float32)

                    word_id = vocab.word_to_id(word)
                    if word_id != vocab.unk_id:
                        glove_mat[word_id, :] = vect
                        num += 1
                    bar.update(idx + 1)
            np.savez(npzfilename, matrix=glove_mat)
        print("Glove loaded: mat %s, vocab size %d" % (glove_mat.shape, num))

    return glove_mat


if __name__ == "__main__":
    # text_seqs, vocab = load_data(config.wiki_train_data_path, config.wiki_vocab_path, config.wiki_train_processed_path, column="text", force_process=True)
    # text_seqs, vocab = load_data(config.arxiv_train_data_path, config.arxiv_vocab_path, config.arxiv_train_processed_path, column="abstract", force_process=True)

    # vocab = build_vocabulary_from_full_corpus(config.wiki_full_data_path, config.wiki_vocab_path, column="text", force_process=True)
    # vocab = build_vocabulary_from_full_corpus(config.arxiv_full_data_path, config.arxiv_vocab_path, column="abstract", force_process=True)
    vocab = build_vocabulary_from_full_corpus(config.zhang15_dbpedia_full_data_path, config.zhang15_dbpedia_vocab_path,
                                              column="text", force_process=False)
    # vocab = build_vocabulary_from_full_corpus(config.zhang15_yahoo_full_data_path, config.zhang15_yahoo_vocab_path, column=["question_title", "question_content", "best_answer"], force_process=False)
    # vocab = build_vocabulary_from_full_corpus(config.chen14_full_data_path, config.chen14_vocab_path, column="text", min_word_count=1, force_process=False)
    # vocab = build_vocabulary_from_full_corpus(config.news20_full_data_path, config.news20_vocab_path, column="text", min_word_count=1, force_process=False)

    # text_seqs = load_data_from_text_given_vocab(
    #     config.zhang15_dbpedia_test_path, vocab, config.zhang15_dbpedia_test_processed_path,
    #     column="text", force_process=False
    # )

    # text_seqs = load_data_from_text_given_vocab(
    #     config.zhang15_dbpedia_train_path, vocab, config.zhang15_dbpedia_train_processed_path,
    #     column="text", force_process=False
    # )

    # text_seqs = load_data_from_text_given_vocab(
    #     config.zhang15_yahoo_test_path, vocab, config.zhang15_yahoo_test_processed_path,
    #     column=["question_title", "question_content", "best_answer"], force_process=False
    # )

    # print(len(text_seqs))
    # print(text_seqs[0])

    # load_kg_vector(config.kg_vector_data_path)
    # data_class_list = load_data_class(
    #     filename=config.zhang15_dbpedia_train_path,
    #     column="class",
    # )
    # print(data_class_list)

    # class_dict = load_class_dict(
    #     class_file=config.zhang15_dbpedia_class_label_path,
    #     class_code_column="ClassCode",
    #     class_name_column="ConceptNet"
    # )
    # print(class_dict)

    # load_glove_word_vector(config.word_embed_file_path, vocab)

    '''
    vocab = build_vocabulary_from_full_corpus(
        config.chen14_full_data_path, config.chen14_vocab_path, column="text", force_process=False
    )

    glove_mat = load_glove_word_vector(
        config.word_embed_file_path, config.chen14_word_embed_matrix_path, vocab
    )

    class_dict = load_class_dict(
        class_file=config.chen14_class_label_path,
        class_code_column="ClassCode",
        class_name_column="ConceptNet"
    )

    for class_id in class_dict:
        class_label = class_dict[class_id]
        class_label_word_id = vocab.word_to_id(class_label)
        print(class_label, class_label_word_id, np.sum(glove_mat[class_label_word_id]))
    '''

    '''
    wordlist = ["operating-system", "middle-east", "operating_system", "operating", "os"]
    filename = config.word_embed_file_path
    with open(filename, 'r') as f:
        for line in f:
            content = line.replace("\n", "").split(" ")
            word = content[0]
            for w in wordlist:
                if w == word:
                    print(w)
    '''

    pass
