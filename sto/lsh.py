"""
modified from https://github.com/ekzhu/datasketch

Thanks for the repo.
"""

import hashlib
import struct
import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod

_mersenne_prime = (1 << 61) - 1
_max_hash = (1 << 32) - 1
_integration_precision = 0.001


def sha1_hash32(data):
    # https://searchnetworking.techtarget.com/definition/big-endian-and-little-endian
    return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]


def _integration(f, a, b):
    p = _integration_precision
    area = 0.0
    x = a
    while x < b:
        area += f(x+0.5*p)*p
        x += p
    return area, None


def _false_positive_probability(threshold, b, r):
    def _probability(s): return 1 - (1 - s**float(r))**float(b)
    a, err = _integration(_probability, 0.0, threshold)
    return a


def _false_negative_probability(threshold, b, r):
    def _probability(s): return 1 - (1 - (1 - s**float(r))**float(b))
    a, err = _integration(_probability, threshold, 1.0)
    return a


def _optimal_param(
        threshold,
        num_perm,
        false_positive_weight,
        false_negative_weight):
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm+1):
        max_r = int(num_perm / b)
        for r in range(1, max_r+1):
            fp = _false_positive_probability(threshold, b, r)
            fn = _false_negative_probability(threshold, b, r)
            error = fp*false_positive_weight + fn*false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class MinHash:
    def __init__(self, num_perm=128, seed=1, hashfunc=sha1_hash32):
        self.seed = seed
        self.hashfunc = hashfunc
        self.hashvalues = np.ones(
            num_perm, dtype=np.uint64) * _max_hash
        generator = np.random.RandomState(self.seed)
        self.permutations = np.array(
            [(generator.randint(1, _mersenne_prime, dtype=np.uint64),
              generator.randint(0, _mersenne_prime, dtype=np.uint64)
              ) for _ in range(num_perm)
             ], dtype=np.uint64).T

    def update(self, data):
        hv = self.hashfunc(data)
        a, b = self.permutations
        phv = np.bitwise_and((a * hv + b) %
                             _mersenne_prime, np.uint64(_max_hash))
        self.hashvalues = np.minimum(phv, self.hashvalues)

    def jaccard(self, arr: np.array):
        return np.float(
            np.count_nonzero(self.hashvalues == arr
                             ) / len(self.hashvalues))


class LSH:
    def __init__(self, threshold=0.9, num_perm=128, weights=(0.5, 0.5)):
        false_positive_weight, false_negative_weight = weights
        self.b, self.r = _optimal_param(
            threshold, num_perm, false_positive_weight, false_negative_weight)
        self.hashtables = [DictSetStorage() for i in range(self.b)]
        self.hashranges = [(i*self.r, (i+1)*self.r) for i in range(self.b)]
        self.hashkeys = DictListStorage()
        self.hashvalues = {}

    def insert(self, key, minhash):
        Hs = [self._H(minhash.hashvalues[start:end])
              for start, end in self.hashranges]
        self.hashkeys.insert(key, *Hs)
        self.hashvalues[key] = minhash.hashvalues
        for H, hashtable in zip(Hs, self.hashtables):
            hashtable.insert(H, key)

    def query(self, minhash):
        candidates = {}
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            for key in hashtable.get(H):
                if key in candidates:
                    continue
                candidates[key] = self.hashvalues[key]
        return candidates

    def remove(self, key):
        for H, hashtable in zip(self.hashkeys[key], self.hashtables):
            hashtable.remove_val(H, key)
            if not hashtable.get(H):
                hashtable.remove(H)
        self.hashkeys.remove(key)
        del self.hashvalues[key]

    def is_empty(self):
        return any(t.size() == 0 for t in self.hashtables)

    def __contains__(self, key):
        return key in self.hashkeys

    @staticmethod
    def _H(hs):
        return bytes(hs.byteswap().data)


class Storage(ABC):
    def __getitem__(self, key):
        return self.get(key)

    def __delitem__(self, key):
        return self.remove(key)

    def __len__(self):
        return self.size()

    def __iter__(self):
        for key in self.keys():
            yield key

    def __contains__(self, item):
        return self.has_key(item)

    @abstractmethod
    def keys(self):
        return []

    @abstractmethod
    def get(self, key):
        pass

    @abstractmethod
    def insert(self, key, *vals, **kwargs):
        pass

    @abstractmethod
    def remove(self, *keys):
        pass

    @abstractmethod
    def remove_val(self, key, val):
        pass

    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def itemcounts(self, **kwargs):
        pass

    @abstractmethod
    def has_key(self, key):
        pass


class DictListStorage(Storage):
    def __init__(self):
        self._dict = defaultdict(list)

    def keys(self):
        return self._dict.keys()

    def get(self, key):
        return self._dict.get(key, [])

    def remove(self, *keys):
        for key in keys:
            del self._dict[key]

    def remove_val(self, key, val):
        self._dict[key].remove(val)

    def insert(self, key, *vals, **kwargs):
        self._dict[key].extend(vals)

    def size(self):
        return len(self._dict)

    def itemcounts(self, **kwargs):
        return {k: len(v) for k, v in self._dict.items()}

    def has_key(self, key):
        return key in self._dict


class DictSetStorage(DictListStorage):
    def __init__(self):
        self._dict = defaultdict(set)

    def get(self, key):
        return self._dict.get(key, set())

    def insert(self, key, *vals, **kwargs):
        self._dict[key].update(vals)


if __name__ == '__main__':
    pass
