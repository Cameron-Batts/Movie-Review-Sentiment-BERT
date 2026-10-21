# Movie Review Sentiment Classification with BERT

**Portfolio presentation:** Cameron Batts  
**Contribution context:** Collaborative project; individual responsibilities were not specified in the supplied notebook. Shared results are not represented as sole authorship.

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
Syntax checked; not rerun. Original results were retained. Dependencies and source revisions are unpinned. Only one seed/split was reported. Prior training overlap for the external checkpoint has not been audited. Stopword removal may remove negation. The new binary head is reinitialized because the original checkpoint head has three outputs; compatible encoder weights are retained.

## Sources and Contributions
Dataset: https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes . Checkpoint identifier: `MarieAngeA13/Sentiment-Analysis-BERT`. Source identifiers and applicable attribution are retained. Observe dataset/model terms. This is a collaborative analysis presented by Cameron Batts; the original notebook does not specify individual contributions. Raw reviews and saved person-name excerpts are omitted.

Portfolio: https://cameron-batts.github.io  
GitHub: https://github.com/Cameron-Batts
