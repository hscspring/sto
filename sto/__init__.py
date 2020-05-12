from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List
import re
import hashlib
import pickle

import pnlp
import dawg
from datasketch import MinHash, MinHashLSHEnsemble


try:
    from ngram.ngram import ngrams
except Exception as e:
    ngrams = (lambda lst, m, n: [
        "".join(lst[i:j])
        for i in range(len(lst))
        for j in range(i+m, min(len(lst), i+n)+1)])


def dump_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


class Tokenizer:

    def __init__(self, lang: str = 'zh'):
        if lang == "zh":
            try:
                from cppjieba import Jieba
                jieba = Jieba()
            except Exception as e:
                import jieba
            self.tokenizer = jieba.cut
        else:
            self.tokenizer = (lambda s: s.split())

    def __call__(self, text: str):
        return list(self.tokenizer(text))


class Sto:

    def __init__(self,
                 value_format: str = 'h',
                 threshold: float = 0.8,
                 num_perm: int = 128,
                 num_part: int = 32,
                 tokenizer: Tokenizer = Tokenizer('zh')):
        self.value_format = value_format
        self.threshold = threshold
        self.num_perm = num_perm
        self.num_part = num_part
        self.tokenizer = tokenizer
        self.lsh = MinHashLSHEnsemble(
            threshold=self.threshold,
            num_perm=self.num_perm)
        self.record_dawg = dawg.RecordDAWG(self.value_format)

    def __check_get_store_path(self, data_path: Path):
        pnlp.check_dir(data_path)
        lsh_path = Path(data_path) / "lsh.pickle"
        dawg_path = Path(data_path) / "values.dawg"
        return lsh_path, dawg_path

    def load(self, data_path: Path):
        lsh_path, dawg_path = self.__check_get_store_path(data_path)
        if lsh_path.exists():
            self.lsh = load_pickle(lsh_path)
        else:
            raise ValueError("lsh pickle: {} not exist.".format(lsh_path))
        if dawg_path.exists():
            self.record_dawg.load(str(dawg_path))
        else:
            raise ValueError("dawg file: {} not exist.".format(dawg_path))

    def store(self, data_path: Path):
        lsh_path, dawg_path = self.__check_get_store_path(data_path)
        dump_pickle(lsh_path, self.lsh)
        self.record_dawg.save(dawg_path)

    def __check_value_format(self, val: tuple):
        if len(val) != len(self.value_format):
            raise ValueError(
                "value format {} does not match the value {}".format(
                    self.value_format, val))

    def add(self, text_list: List[str], value_list: List[tuple]):
        len_text = len(text_list)
        len_value = len(value_list)
        assert len_text == len_value
        data = {}
        entries = []
        for i, text in enumerate(text_list):
            entry = self.text_to_lsh_entry(text)
            key = entry[0]
            if key in data:
                continue
            value = value_list[i]
            self.__check_value_format(value)
            data[key] = value
            entries.append(entry)
        self.lsh.index(entries)
        self.record_dawg = dawg.RecordDAWG(
            self.value_format, data.items())

    def query(self, text: str):
        key, mh, length = self.text_to_lsh_entry(text)
        if key in self.record_dawg:
            return self.record_dawg.get(key)[0]

        for sim_key in self.lsh.query(mh, length):
            return self.record_dawg.get(sim_key)[0]
        else:
            return

    def text_to_lsh_entry(self, text: str):
        words = self.tokenizer(text)
        bigrams = list(ngrams(words, 1, 2))
        wset = set(bigrams)
        mh = MinHash(num_perm=self.num_perm)
        for w in wset:
            mh.update(w.encode('utf8'))
        unicode_hash = hashlib.sha1(text.encode("utf8")).hexdigest()
        return (unicode_hash, mh, len(wset))

    def __getitem__(self, key: str):
        return self.query(key)

    def __setitem__(self, key: str, value: tuple):
        raise NotImplementedError

    def __contains__(self, key: str):
        if self.query(key):
            return True
        return False

    def __len__(self):
        return len(self.record_dawg.keys())


if __name__ == '__main__':
    text_list = [
        "我爱你，你爱我。",
        "我爱她，她爱我。",
        "我爱北京天安门。"
    ]
    value_list = [(2, ), (1, ), (0, )]
    st = Sto(threshold=0.4)

    print(st.threshold)
    st.add(text_list, value_list)
    print("all keys:", st.record_dawg.keys())

    _, m0, _ = st.text_to_lsh_entry(text_list[0])
    _, m1, _ = st.text_to_lsh_entry(text_list[1])
    _, m2, _ = st.text_to_lsh_entry(text_list[2])

    _, m3, _ = st.text_to_lsh_entry("我在天安门。")
    _, m4, _ = st.text_to_lsh_entry("他爱北京天安门。")

    print("m0->m1", m0.jaccard(m1))
    print("m0->m2", m0.jaccard(m2))
    print("m1->m2", m1.jaccard(m2))
    print("m2->m3", m2.jaccard(m3))
    print("m2->m4", m2.jaccard(m4))

    print(st.query("我在天安门。"))
    print(st.query("他爱北京天安门。"))
