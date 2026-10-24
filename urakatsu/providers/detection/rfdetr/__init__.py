"""RF-DETR support for the layout checkpoint: config, processor, and network loading.

The network is the upstream ``rfdetr`` package (pinned to 1.7.0, the version the checkpoint's
``load_model.py`` loads strictly). Modules here import Torch and are only loaded by the provider's
``_load``, so listing providers never imports Torch.
"""
