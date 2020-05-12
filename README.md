# Similar Text Only Run Once

## Install

```bash
$ pip install -r requirements.txt
$ python setup.py install
```

## Usage

```python
from sto import Sto

st = Sto()
# Store the model result
value_list = []
for text in text_list:
    # r1, r2, ... should be int, just easy to store
    r1 = model1(text)
    r2 = model2(text)
    # should be a tuple
    values = (r1, r2)
    value_list.append(values)
st.store(text_list, value_list)

# Query if the given text LSH is similar (of course the same) to the stored text.
values = st.query(text)
```

注意：使用 add 批量添加时只会去重完本完全一致的，不会用相似度去重。

如果需要用 Cython 版的 Ngram，可以在 ngram 目录下编译：

```bash
$ python setup.py build_ext --inplace
```

如果需要用 cppjieba 分词，可以直接安装：

```bash
$ pip install cppjieba
```

