# Similar Text run only Once

## Install

```bash
$ pip install -r requirements.txt
$ python setup.py install
```

## Usage

```python
from sto import Sto, Tokenizer

st = Sto(value_format='hh', # default 'h', means a short integer value.
         threshold=0.8, # default 0.8, similarity threshold
         num_perm=128,  # default 128
         num_part=32,   # default 32
         tokenizer=Tokenizer('zh') # default Tokenizer('zh')
        )
# Store the model result
value_list = []
for text in text_list:
    # r1, r2, ... should be int, just easy to store
    r1 = model1(text)
    r2 = model2(text)
    # should be a tuple(short int, short int), this is what the format 'hh' means.
    values = (r1, r2)
    value_list.append(values)
st.store(text_list, value_list)

# Query if the given text LSH is similar (of course the same) to the stored text.
values = st.query(text)
```

注意：

- 使用 add 批量添加时只会去重完本完全一致的，不会用相似度去重。
- value format: [struct — Interpret bytes as packed binary data — Python 3.8.3rc1 documentation](https://docs.python.org/3/library/struct.html#format-strings)
- threshold, num perm and num part: [MinHash LSH Ensemble — datasketch 1.0.0 documentation](http://ekzhu.com/datasketch/lshensemble.html)

如果需要用 Cython 版的 Ngram，可以在 ngram 目录下编译：

```bash
$ python setup.py build_ext --inplace
```

如果需要用 cppjieba 分词，可以直接安装：

```bash
$ pip install cppjieba
```

