"""Running vision models on another machine.

Two ways in:

- The ``remote`` runtime: any detection, OCR or inpainting provider can run on a Urakatsu
  worker (``urakatsu worker``) somewhere else, such as a notebook or a GPU box. The worker
  loads the same provider class on its own runtime, so pre- and post-processing are the same
  code and the results match a local run. The client is ``client.RemoteModel``, the server is
  ``worker.create_app``, and ``wire`` is the encoding both share.
- An attached server: providers that already run as llama-server or sd-server (PaddleOCR-VL,
  FLUX.2 Klein, RORem) can use one someone else started (``server_url`` in their settings).
  See ``urakatsu.runtimes.server_process.AttachedServer``.
"""

REMOTE = "remote"
# Bumped when a request or reply changes shape; the client refuses a worker that speaks another.
PROTOCOL = 2
