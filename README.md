# Movie Review Sentiment Classification with BERT

**Portfolio presentation:** Cameron Batts  
**Contribution context:** Collaborative project; individual responsibilities were not specified in the supplied notebook. Shared results are not represented as sole authorship.

## Validation update - October 22, 2026

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Naive Bayes | 0.7777 | 0.7777 | 0.7777 |
| Logistic regression | 0.7636 | 0.7636 | 0.7636 |

The seven traditional-classifier cells reproduced the historical rounded results and the 8,530/1,066/1,066 split. BERT training was not run on this CPU environment. The MPS/CUDA workflow now fixes dataset/model revisions, sets the seed before binary-head initialization, limits checkpoint retention and exports aggregate results. Binary-head and Trainer initialization passed a CPU setup check; accelerator execution and the Colab launcher remain untested. The historical team BERT result of 84.80% remains separate and unverified by a new training run.

[Aggregate validation record](reports/validation_results.json). [Repeatable validation script](validation/rerun_analysis.py). Run `python validation/rerun_analysis.py` from the project root. Install `requirements-validation.txt` first; source CSVs stay local.

For training on your Mac, use the main notebook; it selects MPS or CUDA and stops before BERT on CPU. Run all cells and return `reports/gpu_rerun_metrics.json`. A separate [Colab launcher](https://colab.research.google.com/github/Cameron-Batts/Movie-Review-Sentiment-BERT/blob/main/notebooks/movie_review_sentiment_colab.ipynb) is also prepared; it has not been executed here.

## Business Problem
Compare methods for classifying movie-review sentiment to support text triage and aggregate feedback analysis.

## Project Approach
- Use predefined Rotten Tomatoes splits: 8,530 train, 1,066 validation and 1,066 test reviews.
- Fit an 83,753-feature unigram/bigram vocabulary on training text only.
- Compare Multinomial Naive Bayes and logistic regression against fully fine-tuned BERT.
- Replace the source checkpoint's three-class head with a two-class head.
- Train BERT for two epochs with learning rate 2e-5, weight decay 0.01 and seed 42; choose the checkpoint with lowest validation loss.

## Original Results
| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Naive Bayes | 77.77% | 0.7777 |
| Logistic regression | 76.36% | 0.7636 |
| Fine-tuned BERT | 84.80% | 0.8477 |

BERT improved reported accuracy by approximately 7.04 percentage points over Naive Bayes and 8.44 points over logistic regression. Positive-class recall was 0.8968 versus 0.7992 for negative reviews. No production outcome or performance on newly collected reviews was measured.

## Run
```bash
python -m pip install -r requirements.txt
jupyter lab notebooks/movie_review_sentiment_bert.ipynb
```
Dataset and model weights download at runtime. GPU acceleration is recommended; the reported run used a Tesla T4. Training checkpoints require disk space and are not bundled.

## Files
- `notebooks/movie_review_sentiment_bert.ipynb`: cleaned team workflow and retained numerical outputs.
- `reports/movie_review_sentiment_bert.html`: readable analysis report.
- `reports/model_comparison.csv`: original metrics.
- `requirements.txt`: dependencies.
- `portfolio_card.html`: matching portfolio card.

## Validation and Limitations
Traditional baselines rerun; BERT results remain historical. Fixed dataset/model revisions and an accelerator export are now configured, with package versions recorded. Accelerator training remains pending. Only one seed/split was reported. Prior training overlap for the external checkpoint has not been audited. Stopword removal may remove negation. The new binary head is reinitialized because the original checkpoint head has three outputs; compatible encoder weights are retained.

## Sources and Contributions
Dataset: https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes . Checkpoint identifier: `MarieAngeA13/Sentiment-Analysis-BERT`. Source identifiers and applicable attribution are retained. Observe dataset/model terms. This is a collaborative analysis presented by Cameron Batts; the original notebook does not specify individual contributions. Raw reviews and saved person-name excerpts are omitted.

Portfolio: https://cameron-batts.github.io  
GitHub: https://github.com/Cameron-Batts
