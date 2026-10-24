"""Comic text detector (CTD) support: network, pre/post-processing, and the DBNet line utilities.

``network`` imports Torch and is only loaded by the provider's ``_load``. ``processor`` is numpy
only at import time; its torch-using functions import Torch when called. A DBNet ResNet34 detector
reuses ``processor.extract_line_polygons``, ``processor.group_text_lines`` and
``processor.refine_mask``.
"""
