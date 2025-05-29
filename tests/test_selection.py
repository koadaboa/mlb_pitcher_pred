import pandas as pd
import numpy as np

from src.features.selection import _calculate_vif


def test_calculate_vif_with_nan_and_inf() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, np.inf],
            "b": [np.inf, 1.0, 2.0, np.nan],
        }
    )
    result = _calculate_vif(df)
    assert list(result.index) == ["a", "b"]
    assert result.isna().all()

