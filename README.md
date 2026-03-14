# Speech Transformer Benchmark

Lightweight benchmarking framework for evaluating transformer-based speech recognition models.

This project evaluates the performance of Whisper speech models on the LibriSpeech dataset and analyzes trade-offs between transcription accuracy and inference latency.

## Features

- Evaluation of multiple Whisper model variants
- Word Error Rate (WER) computation
- Inference latency benchmarking
- Automated evaluation pipeline
- Visualization of performance metrics

## Dataset

Experiments are conducted on the LibriSpeech dataset.

## Models

The benchmark currently evaluates:

- Whisper Tiny
- Whisper Base
- Whisper Small

## Evaluation Metrics

- Word Error Rate (WER)
- Inference Latency

## Repository Structure

src/ → evaluation pipeline
notebooks/ → experiment notebook
results/ → experiment outputs
figures/ → plots used in report
report/ → research draft

## Installation
pip install -r requirements.txt


## Running the Benchmark
python src/evaluate.py


## Results

Example results include:

- WER comparison across model sizes
- latency measurements

## Example Results

### Word Error Rate

![WER](figures/wer.png)

### Latency

![Latency](figures/la.png)

## Research Report

The draft report describing the experiments is available in:

report/speech_transformer_benchmark.pdf


## Future Work

- noise robustness experiments
- hardware-aware benchmarking
- speech model comparison

## Citation

If you use this work, please cite:

@misc{speech_benchmark_2026,
  title={Benchmarking Transformer-Based Speech Recognition Models},
  author={Your Name},
  year={2026},
  note={Research Report},
}

## Author

Sara Ameli
