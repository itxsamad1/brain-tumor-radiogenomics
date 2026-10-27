import pandas as pd

from btlab.data.registry import assert_unique_patients, hash_split_table


def test_hash_stable():
    df = pd.DataFrame({"patient_id": ["a", "b"], "split": ["train", "test"]})
    assert hash_split_table(df) == hash_split_table(df.copy())


def test_unique_patients_ok():
    df = pd.DataFrame({"patient_id": ["p1", "p2"]})
    assert_unique_patients(df)
