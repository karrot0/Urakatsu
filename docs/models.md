# Models

Urakatsu's default models are downloaded on first use, pinned to exact revisions. Everything
else comes from the Models page, a Hugging Face catalog rated for your machine.

## Defaults

| Stage | Default |
| --- | --- |
| Detection | RF-DETR layout detection (`koharu-layout-rfdetr-seg-2xl`) |
| OCR | PaddleOCR-VL 1.6 |
| Translation | a local GGUF model run by llama.cpp |
| Inpainting | LaMa |
| Lettering | YuzuMarker Font Detection (`yuzumarker-font-detection`, needs PyTorch) |

Weights go to the models folder: `URAKATSU_HOME/models` by default, or the folder chosen in
Settings, then Pipeline, then Storage (also offered by the Windows setup).

## The Models page

Only the built-in defaults are named in code. The Models page, like LM Studio's Discover,
searches Hugging Face per runtime (llama.cpp GGUF, stable-diffusion.cpp, PyTorch, ONNX), and
for every repository:

- asks each provider whether it can use it (`Provider.accepts(facts)`), from the repo's GGUF
  architecture, `config.json`, files and tags;
- rates every quantization against this machine (full GPU, partial offload, CPU only, or too
  large) from the file sizes, the KV cache for the stage's context (read from the GGUF header
  with ranged requests) and the free VRAM and RAM, and recommends the largest one that fits;
- checks the installed runtime release against the architectures and families it supports,
  read from that release's own sources.

Installed models are recorded in the models folder's `installed.json`, with each file's role.
Any stage can switch to one in Settings, or with "Use for" on the Installed tab. PyTorch models
must ship `.safetensors` weights; pickle checkpoints are never loaded.

Catalog answers (searches, model details, GGUF headers, model cards) are kept in
`URAKATSU_HOME/cache`, so the Models page opens instantly after a restart and refreshes in the
background.

## Diffusion models for inpainting

Any stable-diffusion.cpp image model can inpaint. `sd-server` ships a recipe per model
(`examples/server/recipes/*.json`: component slots, sampler defaults, features), read at the
installed release (`urakatsu/catalog/recipes.py`):

- A repo matches a recipe by its author repo or name. Every slot besides the model (text
  encoder, VAE, tokenizer, projector) is taken from the repo itself, or from the companion repos
  the family's `docs/<family>.md` links, and downloaded with it.
- The Diffusion model inpainter (`providers/inpainting/diffusion_recipe.py`) runs whichever one
  is installed. Editing models (recipes with `ref_images`, such as Qwen Image 2.1) get the crop
  as a reference and an instruction to remove the text; the others repaint the masked area.
- Video models, and recipes that need an adapter input (ControlNet, IP-Adapter), are not offered.
- A repo that only holds a text encoder (its GGUF architecture is a language model llama.cpp
  runs) is marked as such and is not installed as an image model.

## Lettering styles

A manga letters each kind of line in its own kind of typeface: dialogue in antique, narration in
mincho, shouts in a heavy gothic, thoughts in a rounded gothic. The lettering stage reads which
one each line is in and gives its translation the matching style, as Koharu 0.100 does.

- It runs after text recognition, in any run that reads or translates text. **Run Lettering**
  in the stage menu runs it alone, so pages that are already detected and read get styles
  without being detected again. Settings, then Typesetting turns it off.
- The font reader names the likeliest of 6,150 Chinese, Japanese and Korean fonts for five crops
  of each line, and their probabilities are summed into 19 styles. A line takes a style only
  when the reader is sure of it; otherwise it is set as dialogue. A book's dialogue keeps one
  face from balloon to balloon.
- A line is set bold when its original's strokes are 1.3 times as thick as the book's
  dialogue, as in Koharu. The weights of the fonts the reader named guard that: a line whose
  fonts read lighter than the dialogue's is not bold, however thick its box measured (a
  balloon's outline or line art inside it), and they decide when no stroke could be measured.
- Settings, then Typesetting lists each style's font, weight and slant per script (Latin,
  Cyrillic, Greek, Chinese and Japanese), with **Restore defaults**. Changing a style's font
  restyles every text of that style. The type panel's **Lettering style** menu sets one text's
  style by hand; a font, weight or slant set on a text wins over its style's.
- The reader is trained on Chinese, Japanese and Korean lettering; what it makes of other
  scripts is not to be relied on. Without PyTorch a run leaves the stage out.

The rules and the measurements behind them are in `urakatsu/pipeline/lettering.py`, the style
table in `urakatsu/core/styles.py`.

## Hosted translation

Hosted providers (Claude, OpenAI, DeepL and others) need an API key: set it in Settings, then
Providers.

## Page storage

Imported and downloaded pages are stored as they arrive (JPEG, PNG or WebP).
Settings, then Pipeline, then Storage can store PNG pages as lossless WebP instead: about 15%
smaller, with identical pixels.
