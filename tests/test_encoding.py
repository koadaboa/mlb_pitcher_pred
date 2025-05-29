import pandas as pd
from src.features.encoding import mean_target_encode


def test_mean_target_encode_fit_transform() -> None:
    df = pd.DataFrame({
        'cat': ['a', 'a', 'b', 'b'],
        'y': [1, 3, 2, 4],
    })
    encoded, mapping = mean_target_encode(df, ['cat'], 'y')
    assert 'cat_enc' in encoded.columns
    # a -> (1 + 3)/2 = 2, b -> (2 + 4)/2 = 3
    assert encoded['cat_enc'].tolist() == [2.0, 2.0, 3.0, 3.0]
    assert 'cat' in mapping


def test_mean_target_encode_transform() -> None:
    train = pd.DataFrame({'cat': ['a', 'b'], 'y': [1, 3]})
    _, mapping = mean_target_encode(train, ['cat'], 'y')
    test = pd.DataFrame({'cat': ['a', 'b', 'c']})
    encoded, _ = mean_target_encode(test, ['cat'], mapping=mapping)
    assert encoded['cat_enc'].tolist() == [1.0, 3.0, float('nan')]

