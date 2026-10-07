"""Statistics and ML over the stored data (Sections 7.1-7.6)."""

import logging

# pandas_ta returns `None` for a series shorter than the indicator's window, and
# says so at WARNING. Both callers here turn that `None` into NaN on purpose
# (`technical._assign_indicator`, `forecasting.build_features`), so it is a
# designed outcome, not a fault -- and the forecasting walk-forward's growing
# windows made it 429 lines of a weekly log (2026-10-05), second only to the
# per-ticker key warnings finding 37 removed. This one logger emits nothing
# else; the rest of the library keeps its level.
logging.getLogger("pandas_ta_classic.utils._core").setLevel(logging.ERROR)
