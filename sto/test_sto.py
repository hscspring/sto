import pytest
from . import Sto

from pathlib import Path


@pytest.fixture
def get_normal_data():
    text_list = [
        "我爱你，你爱我。",
        "我爱她，她爱我。",
        "我爱北京天安门。"
    ]
    value_list = [(1, ), (1, ), (0, )]
    return text_list, value_list


@pytest.fixture
def get_repeat_data():
    text_list = [
        "我爱她，她爱我。",
        "我爱她，她爱我。",
        "我爱北京天安门。"
    ]
    value_list = [(1, ), (1, ), (0, )]
    return text_list, value_list


@pytest.fixture
def get_similar_data():
    text_list = [
        "我爱她，她爱我。",
        "她爱北京天安门。",
        "我爱北京天安门。"
    ]
    value_list = [(1, ), (1, ), (0, )]
    return text_list, value_list


@pytest.fixture
def get_normal_multi_values_data():
    text_list = [
        "我爱你，你爱我。",
        "我爱她，她爱我。",
        "我爱北京天安门。"
    ]
    value_list = [(1, 10), (1, 9), (0, 8)]
    return text_list, value_list


def test_sto_add_normal(get_normal_data):
    text_list, value_list = get_normal_data
    st = Sto()
    st.add(text_list, value_list)
    assert len(st.record_dawg.keys()) == 3
    assert not st.lsh.is_empty()


def test_sto_add_repeat(get_repeat_data):
    text_list, value_list = get_repeat_data
    st = Sto()
    st.add(text_list, value_list)
    assert not st.lsh.is_empty()
    assert len(st.record_dawg.keys()) == 2


def test_sto_add_similar(get_similar_data):
    """
    we do not check when batch adding.
    """
    text_list, value_list = get_similar_data
    st = Sto()
    st.add(text_list, value_list)
    assert not st.lsh.is_empty()
    assert len(st.record_dawg.keys()) == 3


def test_sto_add_multi_values(get_normal_multi_values_data):
    text_list, value_list = get_normal_multi_values_data
    st = Sto(value_format='hh')
    st.add(text_list, value_list)
    assert not st.lsh.is_empty()
    assert len(st.record_dawg.keys()) == 3


def test_query_normal(get_normal_multi_values_data):
    text_list, value_list = get_normal_multi_values_data
    st = Sto(value_format='hh')
    st.add(text_list, value_list)
    text = "我爱北京天安门。"
    values = st.query(text)
    assert values == (0, 8)


def test_query_similar(get_normal_multi_values_data):
    text_list, value_list = get_normal_multi_values_data
    st = Sto(value_format='hh')
    st.add(text_list, value_list)
    text = "她爱北京天安门。"
    values = st.query(text)
    assert values == (0, 8)


def test_query_similar_with_threshold(get_normal_multi_values_data):
    text_list, value_list = get_normal_multi_values_data
    st = Sto(value_format='hh', threshold=0.3)
    st.add(text_list, value_list)
    text = "我在天安门。"
    values = st.query(text)
    assert values == (0, 8)


def test_sto_store(get_normal_multi_values_data):
    data_path = "./data"
    text_list, value_list = get_normal_multi_values_data
    st = Sto(value_format='hh')
    st.add(text_list, value_list)
    st.store(data_path)
    lsh_path = Path(data_path) / "lsh.pickle"
    dawg_path = Path(data_path) / "values.dawg"
    assert lsh_path.exists()
    assert dawg_path.exists()


def test_sto_load():
    st = Sto(value_format='hh')
    st.load("./data")
    assert not st.lsh.is_empty()
    assert len(st.record_dawg.keys()) == 3


def test_sto_get(get_normal_multi_values_data):
    text_list, value_list = get_normal_multi_values_data
    st = Sto(value_format='hh')
    st.add(text_list, value_list)
    assert st["我爱你，你爱我。"] == (1, 10)


def test_sto_contain_similar(get_normal_multi_values_data):
    text_list, value_list = get_normal_multi_values_data
    st = Sto(value_format='hh')
    st.add(text_list, value_list)
    assert "他爱北京天安门。" in st


def test_sto_len(get_normal_multi_values_data):
    text_list, value_list = get_normal_multi_values_data
    st = Sto(value_format='hh')
    st.add(text_list, value_list)
    assert len(st) == 3
