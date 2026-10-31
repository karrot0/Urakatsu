## [0.2.0](https://github.com/karrot0/Urakatsu/compare/v0.1.0..v0.2.0) - 2026-10-31

### ⛰️  Features

- *(canvas)* Improve moving text layers with live text movement ([#3](https://github.com/karrot0/Urakatsu/issues/3)) - ([2b49b16](https://github.com/karrot0/Urakatsu/commit/2b49b16c2e68848d5bf990e2191aa2063c95dbf7))
- *(release)* Implement automated release process with release PR and changelog generation - ([c45cce7](https://github.com/karrot0/Urakatsu/commit/c45cce748dda61236bb80b452371993670668cb3))
- Add knowledge management for translation context - ([8d2c654](https://github.com/karrot0/Urakatsu/commit/8d2c65495b31873854855fd528bd098dcc4e077a))
- Implement network meter for download progress and enhance GPU setup feedback - ([07e7621](https://github.com/karrot0/Urakatsu/commit/07e762134d0b5dd13047112e77a8705dc4004ea2))
- Enhance sound effects processing with forced redraw capabilities and improved layer management - ([6589ea9](https://github.com/karrot0/Urakatsu/commit/6589ea95fab560365ea122bcaa03f6fdabe7d783))
- Add support for PTX-only GPUs in preferred variants logic and new test for CUDA build selection - ([03b2a75](https://github.com/karrot0/Urakatsu/commit/03b2a758565399cd36fd23fde0497f1c53793766))
- Workers using Kaggle and Colab - ([cd78abe](https://github.com/karrot0/Urakatsu/commit/cd78abea5528e71fe8a197a5a1bd640770b7aad2))
- Implement Charspot text detection provider and corresponding OCR functionality - ([c167b26](https://github.com/karrot0/Urakatsu/commit/c167b26f5f62d0af2a67225937008065abe402bf))
- Enhance sound effect detection and suppress doubled lettering logic - ([c0d69f0](https://github.com/karrot0/Urakatsu/commit/c0d69f04c82d2d146f52e580fcc567b2f88c8e1b))
- Add Python startup check and corresponding tests - ([15d252f](https://github.com/karrot0/Urakatsu/commit/15d252fce1ca3e038e838ce9530df96e852d57f7))
- Tweak light text detection and adjust halo rim detection logic - ([373ee5b](https://github.com/karrot0/Urakatsu/commit/373ee5bd6aae53cfb2879a0a82ae0c02e9a436f1))
- Add local translator tests and implement strftime_now for chat templates - ([70665b4](https://github.com/karrot0/Urakatsu/commit/70665b4f8db4904beea4e2cbbb2d40c0e3fa8164))
- Add mobile support and enhance UI components for better responsiveness - ([5dbd3f0](https://github.com/karrot0/Urakatsu/commit/5dbd3f0ef64a5cc8122b6fb2ce441e9652d86216))
- Webtoon - ([5584c63](https://github.com/karrot0/Urakatsu/commit/5584c6393ad6e4cb523f4bc3be4fafe39ecc34a1))
- Add steps and guidance scale settings to RoremMixed and Flux2Klein providers; update tests for parameter validation - ([8e4f61c](https://github.com/karrot0/Urakatsu/commit/8e4f61c1b5050f61c0b6272ad557830ab3ff9911))
- Support for Linux based Stable Diffusion Cuda - ([2c3ec70](https://github.com/karrot0/Urakatsu/commit/2c3ec70273f891637c11e5e9f9f522f20a772e9e))
- Add macOS x64 variant support and update preferred variants logic - ([da78bd6](https://github.com/karrot0/Urakatsu/commit/da78bd622b4b192344a83b21b4ccc0b0d6f1132b))
- Add Nano Banana 2.1 inpainting provider with Gemini and Urakatsu Cloud support - ([171b7b2](https://github.com/karrot0/Urakatsu/commit/171b7b2964871bcf407ba57c22105ca1fbc4239a))

### 🐛 Bug Fixes

- Add GTK folder picker for Linux and update file dialog usage - ([c9515c3](https://github.com/karrot0/Urakatsu/commit/c9515c314420bc961156f93a66f316f9069f8618))
- ONNX models ran on CPU because preload_dlls was called with an unknown argument - ([ecca7ec](https://github.com/karrot0/Urakatsu/commit/ecca7ec9693ec3fc917679eaaae5449d68017df4))
- Set edits attribute to True for consistent cropping behavior in Flux2KleinProcessor - ([3b7cc02](https://github.com/karrot0/Urakatsu/commit/3b7cc02a04155d922af124e7504d80d1f8ba66f2))
- Stroke and font color not being detected and rendered correctly - ([ae04262](https://github.com/karrot0/Urakatsu/commit/ae04262bb054ff5a2ba3f5e5a33e2fc9e407a72d))
- Tighnted crop for editing models. On Qwen 2.1, the model failed to remove text because the crop was 128px which included other text. Making the model remove the other text instead of the original. - ([c05dd44](https://github.com/karrot0/Urakatsu/commit/c05dd44d315ec47c8ea2288311f543e1bb738fd5))
- Issue with canceling a run only cancelled in-between stages, causing the run to the take the slot and not allowing other runs. - ([1c89fe2](https://github.com/karrot0/Urakatsu/commit/1c89fe2f40707d0de2014d8ba5d20114d254573f))
- Make Flux2 Klein use diffusion, allows to replace it with other Flux models from catalog - ([340e7f8](https://github.com/karrot0/Urakatsu/commit/340e7f81df3427149d017d9dd57d35f1a6883ca1))
- Correct docstring formatting in cloud.py - ([1a529d3](https://github.com/karrot0/Urakatsu/commit/1a529d38c8022c20d3dec9c22e37f2d232c78f48))
- App not opening because it thinks theres already one open - ([cf33aaf](https://github.com/karrot0/Urakatsu/commit/cf33aaf3e6c01781a10e9ed310c456f0a6e49f75))

### 💼 Other

- Perf/text layer drag ([#4](https://github.com/karrot0/Urakatsu/issues/4)) - ([048ecbb](https://github.com/karrot0/Urakatsu/commit/048ecbb099466f81a7b726d86d09effaaa4c0a14))
- Watermark remover - ([1b02512](https://github.com/karrot0/Urakatsu/commit/1b02512df575c1c0fd40cae48c81e65384550629))
- Feat/komga export ([#1](https://github.com/karrot0/Urakatsu/issues/1)) - ([7ff39b2](https://github.com/karrot0/Urakatsu/commit/7ff39b2ff9acbeccb1d302165af52eb5cee07cb6))


## [0.2.0] - 2026-10-25

### ⛰️  Features

- *(automation)* Add Discord Webhook message to automation - ([de6f724](https://github.com/karrot0/Urakatsu/commit/de6f724fcdd6cf0dac6bb806f45aabbe8b491aac))
- Implement remote worker for model serving - ([50129e2](https://github.com/karrot0/Urakatsu/commit/50129e2405f035d685301d96c7fe5dd37d8adde6))
- Add live diffusion previews to job state - ([8504a70](https://github.com/karrot0/Urakatsu/commit/8504a700f738b445744da525903346db07f65732))
- Download queue - ([9c2223d](https://github.com/karrot0/Urakatsu/commit/9c2223db3e4bf3e91d12a978db718287862b4e90))
- Add gradient support to rich text formatting - ([412407b](https://github.com/karrot0/Urakatsu/commit/412407bfe9cb3a3787a7d697107bea560b1fc558))
- Rich Text Formatting & Gradient support - ([a41c7e4](https://github.com/karrot0/Urakatsu/commit/a41c7e4633cdddde506878876a64f95160640ed5))
- Network server - ([4cda39d](https://github.com/karrot0/Urakatsu/commit/4cda39d2df5671bb3e93bc6a7a0e76c24cc68c1d))
- Implement Automations with profiles - ([83ae060](https://github.com/karrot0/Urakatsu/commit/83ae06056ca238bbf1bf420c477940060543abc0))
- Add window controls and update handling for app updates - ([8b668aa](https://github.com/karrot0/Urakatsu/commit/8b668aadd3dbbe515799a932a93f07175c4fc80b))

### 🐛 Bug Fixes

- *(build)* Fix testing for OS - ([59f7436](https://github.com/karrot0/Urakatsu/commit/59f7436ff31d88ece1d5500d3e5e08dfe0566335))
- *(build)* Update workflows - ([f7643ef](https://github.com/karrot0/Urakatsu/commit/f7643efa5f4cfc31a0f737c0cfdd6afed4e2fbd4))
- *(fontsystem)* Enhance font resolution fallback for non-Latin scripts - ([dc73631](https://github.com/karrot0/Urakatsu/commit/dc73631edef830cab0ded6844e9dd609836c4d16))
- *(catalog)* Rate diffusion models by what --auto-fit can stream - ([6d7144b](https://github.com/karrot0/Urakatsu/commit/6d7144b908c3889ab407ed8c111fc5d806c9f5ae))
- Fix(Build) - ([44c0b77](https://github.com/karrot0/Urakatsu/commit/44c0b772b9cac815ab20b219a644dcb31f6eb559))

### 💼 Other

- Add DBNet text detector with ResNet34 backbone and preprocessing - ([944e25a](https://github.com/karrot0/Urakatsu/commit/944e25a57b2039a71d32528bc801310926e31321))
- Implement `urakatsu dev` command for development workflow with service restart on Python changes - ([a1f3052](https://github.com/karrot0/Urakatsu/commit/a1f30521db8f4b5926209870be9a4c0368dee93e))
- Initial Commit - ([3eed64f](https://github.com/karrot0/Urakatsu/commit/3eed64f1dad90c3f25db8b9717bb5da13658b223))

## New Contributors ❤️

* @karrot0 made their first contribution

<!-- generated by git-cliff -->
