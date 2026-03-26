# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**  
Describe whether you used the rule based model, the ML model, or both.  
Example: “I used the rule based model only” or “I compared both models.”

**Your answer:**  
I compared both models.

- Rule based model in `mood_analyzer.py`
- ML model in `ml_experiments.py` (CountVectorizer + LogisticRegression)

**Intended purpose:**  
What is this model trying to do?  
Example: classify short text messages as moods like positive, negative, neutral, or mixed.

**Your answer:**  
Classify short social-style text into one of four mood labels: `positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):**  
For the rule based version, describe the scoring rules you created.  
For the ML version, describe how training works at a high level (no math needed).

**Your answer:**  
Rule based: preprocess text into tokens, add/subtract sentiment score from positive/negative word matches, apply negation and targeted sarcasm/context rules, then map score to label.

ML based: convert text into bag-of-words features using `CountVectorizer`, then train a `LogisticRegression` classifier on `SAMPLE_POSTS` and `TRUE_LABELS`.


## 2. Data

**Dataset description:**  
Summarize how many posts are in `SAMPLE_POSTS` and how you added new ones.

**Your answer:**  
The dataset currently has 20 posts in `SAMPLE_POSTS` with 20 matching labels in `TRUE_LABELS`.
I expanded from the starter set by adding examples with slang, emojis, sarcasm, neutral phrasing, and mixed emotions.

**Labeling process:**  
Explain how you chose labels for your new examples.  
Mention any posts that were hard to label or could have multiple valid labels.

**Your answer:**  
I labeled based on intended sentiment, not just literal keyword polarity.
Hard/debatable examples included:

- "not bad at all" (neutral vs mildly positive)
- "bad but manageable" (negative vs mixed)
- "meh, not great not terrible" (neutral vs mixed)
- "traffic again 😂 love that for me" (literal positive word "love" but negative intent)

**Important characteristics of your dataset:**  
Examples you might include:  

- Contains slang or emojis  
- Includes sarcasm  
- Some posts express mixed feelings  
- Contains short or ambiguous messages

**Your answer:**  
Yes, all of those characteristics are present.

- Slang: `lowkey`, `highkey`, `no cap`, `meh`
- Emojis: `😂`, `💀`
- Sarcasm: traffic and meeting complaints written with positive terms
- Mixed/ambiguous: several `but` sentences and short neutral lines like "it is what it is"

**Possible issues with the dataset:**  
Think about imbalance, ambiguity, or missing kinds of language.

**Your answer:**  
Main issues:

- Small dataset size (20 examples)
- Ambiguous labels for some edge cases
- Limited domain and writing style coverage
- Potential label imbalance
- No held-out test set in the current ML script

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  
Describe the modeling choices you made.  
Examples:  

- How positive and negative words affect score  
- Negation rules you added  
- Weighted words  
- Emoji handling  
- Threshold decisions for labels

**Your answer:**  
Scoring rules used:

- Base scoring: positive token `+1`, negative token `-1`
- Negation handling: `not/never/no/can't/cannot/don't` flips polarity of the next sentiment token
- Context/sarcasm adjustments:
  - if tokens include `love + traffic + (stuck or waiting)`, apply extra negative shift (`-2`)
  - if tokens include `great + another + meeting`, apply extra negative shift (`-2`)
- Label thresholds:
  - if both positive and negative hits are present and `abs(score) <= 1`, label as `mixed`
  - otherwise `score > 0` => `positive`, `score < 0` => `negative`, `score == 0` => `neutral`

**Strengths of this approach:**  
Where does it behave predictably or reasonably well?

**Your answer:**  
Strengths:

- Transparent logic and easy debugging
- Predictable behavior on explicit sentiment words
- Better handling of negation than a naive keyword counter
- Fast runtime and low complexity

**Weaknesses of this approach:**  
Where does it fail?  
Examples: sarcasm, subtlety, mixed moods, unfamiliar slang.

**Your answer:**  
Weaknesses:

- Brittle to unseen phrasing and new slang
- Sarcasm coverage is limited to hardcoded patterns
- Mixed vs neutral is still hard in nuanced sentences
- Vocabulary gaps (for example words like `proud` or `hope`) can cause misses

## 4. How the ML Model Works (if used)

**Features used:**  
Describe the representation.  
Example: “Bag of words using CountVectorizer.”

**Your answer:**  
Bag-of-words text representation with `CountVectorizer`.

**Training data:**  
State that the model trained on `SAMPLE_POSTS` and `TRUE_LABELS`.

**Your answer:**  
The model was trained on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`.

**Training behavior:**  
Did you observe changes in accuracy when you added more examples or changed labels?

**Your answer:**  
Yes. The model changed quickly with the new labeled examples and matched many of the new patterns.
In the current script, ML is evaluated on the same dataset it trains on, so reported accuracy stayed very high.

**Strengths and weaknesses:**  
Strengths might include learning patterns automatically.  
Weaknesses might include overfitting to the training data or picking up spurious cues.

**Your answer:**  
Strengths:

- Learns token-label patterns automatically
- Adapts quickly when new examples are added

Weaknesses:

- Current evaluation is training-set accuracy, so it can be over-optimistic
- Small data means higher risk of overfitting and spurious correlations

## 5. Evaluation

**How you evaluated the model:**  
Both versions can be evaluated on the labeled posts in `dataset.py`.  
Describe what accuracy you observed.

**Your answer:**  
I evaluated both models on the current 20 labeled posts.

- Rule based accuracy: `0.55`
- ML accuracy: `1.00` (on the same dataset used for ML training)

**Examples of correct predictions:**  
Provide 2 or 3 examples and explain why they were correct.

**Your answer:**  
Examples:

- "I am not happy about this" -> `negative` (negation rule flips `happy`)
- "I can't even tell if today was good or bad" -> `mixed` (balanced positive/negative cues)
- "Great, another meeting that could have been an email" -> `negative` (context/sarcasm rule)

**Examples of incorrect predictions:**  
Provide 2 or 3 examples and explain why the model made a mistake.  
If you used both models, show how their failures differed.

**Your answer:**  
Rule-based failure examples:

- "lowkey proud but also stressed about finals" -> predicted `negative`, true `mixed`
- "traffic again 😂 love that for me" -> predicted `positive`, true `negative`
- "No sleep, no coffee, no hope" -> predicted `neutral`, true `negative`

Failure differences:

- Rule based failures mostly come from lexicon gaps and unseen phrasing.
- ML fit this dataset extremely well, but this may not hold on unseen data because it is currently evaluated on training data.

## 6. Limitations

Describe the most important limitations.  
Examples:  

- The dataset is small  
- The model does not generalize to longer posts  
- It cannot detect sarcasm reliably  
- It depends heavily on the words you chose or labeled

**Your answer:**  
Key limitations:

- Small dataset
- Ambiguous labels in some edge cases
- Rule based sarcasm handling is narrow and pattern-specific
- ML score is likely optimistic due to training-set evaluation
- Both models are tuned for short, informal text only

## 7. Ethical Considerations

Discuss any potential impacts of using mood detection in real applications.  
Examples: 

- Misclassifying a message expressing distress  
- Misinterpreting mood for certain language communities  
- Privacy considerations if analyzing personal messages

**Your answer:**  
Potential risks:

- Distress or urgent messages could be misclassified
- Slang/dialect differences could create fairness issues
- Overreliance on noisy labels may lead to harmful decisions
- Personal text analysis requires user consent and careful privacy handling

## 8. Ideas for Improvement

List ways to improve either model.  
Possible directions:  

- Add more labeled data  
- Use TF IDF instead of CountVectorizer  
- Add better preprocessing for emojis or slang  
- Use a small neural network or transformer model  
- Improve the rule based scoring method  
- Add a real test set instead of training accuracy only

**Your answer:**  
Improvement ideas:

- Add more diverse labeled posts (especially sarcasm and mixed sentiment)
- Add a train/dev/test split for fair ML evaluation
- Try `TfidfVectorizer` and compare against bag-of-words counts
- Expand sentiment lexicons with missing words (`proud`, `hopeful`, `chaos`, etc.)
- Improve emoji/slang-specific rule logic
- Track class-level metrics (precision/recall) in addition to overall accuracy
