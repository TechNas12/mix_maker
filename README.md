# 🎵 Mix Maker

**An advanced, asynchronous AI pipeline for generative music creation, variation orchestration, and automated audio stitching.**

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![GenAI](https://img.shields.io/badge/Google_GenAI-Gemini_%7C_Lyria-orange)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Mix Maker is a robust, terminal-based AI application that orchestrates multimodal generative AI models to create and arrange music. It demonstrates production-level patterns for AI engineering, including multi-model pipelines, concurrent asynchronous API interactions, robust error handling with exponential backoff, and programmatic audio manipulation.

## ✨ Key Features (AI Engineering Highlights)

- **Multi-Model Orchestration:** Seamlessly chains models. Uses **Gemini (gemini-3.1-flash-lite)** to expand a base user prompt into `n` diverse creative variations based on a chosen style, which are then passed to **Lyria (lyria-3-pro-preview)** for high-fidelity audio generation.
- **High-Performance Asynchronous Fan-out:** Employs `asyncio` and semaphores to manage concurrent requests to the music generation API, maximizing throughput while respecting rate limits.
- **Robust Resilience & Retry Logic:** Built for unreliable networks and API limits. Implements automated retries with exponential backoff for LLM calls and graceful degradation if individual audio track generations fail.
- **Automated Audio Stitching:** Uses `pydub` to automatically stitch generated tracks together. Supports dynamic audio manipulation including direct concatenation, exact millisecond gap insertions, and smooth crossfading.
- **Beautiful & Responsive CLI:** Features a rich, dynamic terminal interface built with `rich`, providing live progress bars, detailed structured tables, and animated status updates (bypassing event-loop conflicts on Windows).
- **Comprehensive Auditing & Manifests:** Every run generates a timestamped directory containing all raw audio variations, the final stitched track, and a detailed JSON manifest for complete traceability of the generation process.

## 🏗️ Architecture overview

1. **LLM Stage:** Takes a base prompt + style definition and queries Gemini to generate creative prompt variations.
2. **Music Generation Stage:** Fans out the prompt variations concurrently via Google's Lyria model. Downloads the resulting inline audio blobs.
3. **Stitching Stage:** Collects successfully generated tracks and uses `MixMaker` to stitch them together based on user-defined crossfade or gap parameters.

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- Google AI Studio API Key (`GOOGLE_API_KEY`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/mix-maker.git
   cd mix-maker
   ```

2. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_genai_api_key_here
   LLM_KEY=your_google_genai_api_key_here
   MUSIC_KEY=your_google_genai_api_key_here
   ```

3. **Install dependencies:**
   Using `uv`:
   ```bash
   uv sync
   ```
   Or using standard `pip`:
   ```bash
   pip install -e .
   ```

### Usage

Run the main CLI pipeline:

```bash
mixmaker
```

The interactive CLI will prompt you for:
1. **Style:** (e.g., electronic, classical, ambient)
2. **Base Prompt:** A description of the vibe or subject of your music.
3. **Variations:** How many different tracks to generate.

Output files, including individual tracks, the stitched result, and the JSON manifest, will be saved in a timestamped folder under `artifacts/`.

## 🛠️ Tech Stack

- **AI/ML:** `google-genai` (Gemini & Lyria)
- **Audio Processing:** `pydub`
- **Concurrency:** Native Python `asyncio`
- **CLI/UI:** `rich`
- **Package Management:** `hatchling`, `uv`

## 🧠 Design Philosophy

This project is built to showcase clean AI software engineering practices:
- **Separation of Concerns:** Distinct modules for core logic (`MixMaker`), LLM providers (`LLM`), music providers (`MusicGenerator`), and UI (`interface.py`).
- **Resilience:** Defensive programming around external API calls.
- **User Experience:** Even CLI tools deserve a beautiful, informative interface.

---
*Crafted as a demonstration of production-ready AI engineering.*