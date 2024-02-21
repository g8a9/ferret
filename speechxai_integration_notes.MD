# Speech-XAI-Ferret integration notes

**General idea:** as much as SpeechXAI and Ferret has been designed with a similar structure in mind, some common aspects of the two libraries have not been abstracted away in either from the start. A thorough integration work would thus require some refactoring in both libraries, which would take time. Moreover, it's been decided to follow a general principle of decoupling, with the speech functionalities being completely independent (and even optionally installable) from the text ones. Therefore, the first step is just that of **functional integration**, with no attempt of abstracting the common components.

## Evaluation metrics

SpeechXAI provides two new evaluation metrics analogous to some of those already present in Ferret. These may or may not be rewritten as subclasses of the `BaseEvaluator` ABC for consistency with Ferret. **Idea:** for now, just write another ABC specific for speech so that decoupling is guaranteed, even though some overhead/reproduced code is introduced.

# Required actions

- Add mp3 files for white and pink noise into `ferret/explainers/explanation_speech/`. --> DONE
- `ferret/explainers/__init__.py` contains the `BaseExplainer` ABC, which Speech-XAI doesn't seem to have an equivalent for. Should this be included somehow?
- Check how to make the speech part installable separately from the text part (with options like `pip install ferret[all]`, `pip install ferret[text]` or `pip install ferret [speech]`).