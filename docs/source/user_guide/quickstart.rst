.. _quickstart:

**********
Quickstart
**********

Here is a code snipped to show **ferret** integrated with your existing **transformers** models for a text-based task.

.. code-block:: python

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from ferret import Benchmark

    name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)

    bench = Benchmark(model, tokenizer)
    explanations = bench.explain("You look stunning!", target=1)
    evaluations = bench.evaluate_explanations(explanations, target=1)

    bench.show_evaluation_table(evaluations)

The ferret library also streamlines working with audio (speech) data.

.. code-block:: python

    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
    from ferret import SpeechBenchmark, AOPC_Comprehensiveness_Evaluation_Speech

    model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ic")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superbwav2vec2-base-superb-ic")

    speech_benchmark = SpeechBenchmark(model, feature_extractor)
    explanation = speech_benchmark.explain(audio_path=audio_path, methodology='LOO')
    aopc_compr = AOPC_Comprehensiveness_Evaluation_Speech(benchmark.model_helper)
    evaluation_output_c = aopc_compr.compute_evaluation(explanation)

