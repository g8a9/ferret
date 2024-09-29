import pytest
import math
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from ferret import (
    Benchmark,
    LIMEExplainer,
    SHAPExplainer,
    GradientExplainer,
    IntegratedGradientExplainer,
)
from ferret.evaluators.faithfulness_measures import (
    AOPC_Comprehensiveness_Evaluation,
    AOPC_Sufficiency_Evaluation,
    TauLOO_Evaluation,
)
from ferret.evaluators.plausibility_measures import (
    AUPRC_PlausibilityEvaluation,
    Tokenf1_PlausibilityEvaluation,
    TokenIOU_PlausibilityEvaluation,
)
from ferret.evaluators.class_measures import AOPC_Comprehensiveness_Evaluation_by_class
from ferret.modeling.text_helpers import SequenceClassificationHelper

DEFAULT_EXPLAINERS_NUM = 6
DEFAULT_EVALUATORS_NUM = 6
DEFAULT_EVALUATORS_BY_CLASS_NUM = 1

TASK_NAME_MAP = {
    "lvwerra/distilbert-imdb": "text-classification",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli": "nli",
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli": "zero-shot-text-classification",
    "Babelscape/wikineural-multilingual-ner": "ner",
}
explainer_init_extra_args = {
    GradientExplainer: {"multiply_by_inputs": True},
    IntegratedGradientExplainer: {"multiply_by_inputs": True},
}


# ============================================================
# = Fixtures creation to initalize each model and tokenizer  =
# ============================================================
@pytest.fixture(
    scope="module",
    params=[
        ("lvwerra/distilbert-imdb", AutoModelForSequenceClassification),
        (
            "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            AutoModelForSequenceClassification,
        ),
        ("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", AutoModelForSequenceClassification),
        ("Babelscape/wikineural-multilingual-ner", AutoModelForTokenClassification),
    ],
    ids=["textclass", "nli", "zeroshot", "ner"],
)
def model_and_tokenizer(request):
    model_cls = request.param[1]
    model = model_cls.from_pretrained(request.param[0])
    tokenizer = AutoTokenizer.from_pretrained(request.param[0])
    task_name = TASK_NAME_MAP[request.param[0]]

    return model, tokenizer, task_name


@pytest.fixture(scope="module")
def explainer(request, model_and_tokenizer):
    model, tokenizer, task_name = model_and_tokenizer
    kwargs = explainer_init_extra_args.get(request.param, {})
    return request.param(model, tokenizer, task_name=task_name, **kwargs)


@pytest.fixture
def model_tokenizer_ner():
    model = AutoModelForTokenClassification.from_pretrained(
        "Babelscape/wikineural-multilingual-ner"
    )
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
    return model, tokenizer


# ================================================================
# =  Fixture for all fixtures (initialization of the benchmarks) =
# ================================================================
@pytest.fixture
def all_benchmarks(model_and_tokenizer):
    model, tokenizer, task_name = model_and_tokenizer
    return Benchmark(model, tokenizer, task_name=task_name)


# =========
# = Tests =
# =========


# Setup and Initialization Checks
def test_initialization_benchmarks(all_benchmarks):
    assert all_benchmarks.model is not None
    assert all_benchmarks.tokenizer is not None
    assert isinstance(all_benchmarks, Benchmark)
    assert len(all_benchmarks.explainers) == DEFAULT_EXPLAINERS_NUM
    assert len(all_benchmarks.evaluators) == DEFAULT_EVALUATORS_NUM
    assert len(all_benchmarks.class_based_evaluators) == DEFAULT_EVALUATORS_BY_CLASS_NUM


def test_explainer_types(all_benchmarks):
    assert (
        isinstance(e, SHAPExplainer)
        or isinstance(e, LIMEExplainer)
        or isinstance(e, GradientExplainer)
        or isinstance(e, IntegratedGradientExplainer)
        for e in all_benchmarks.explainers
    )


def test_evaluator_types(all_benchmarks):
    expected_evaluator_types = [
        AOPC_Comprehensiveness_Evaluation,
        AOPC_Sufficiency_Evaluation,
        TauLOO_Evaluation,
        AUPRC_PlausibilityEvaluation,
        Tokenf1_PlausibilityEvaluation,
        TokenIOU_PlausibilityEvaluation,
    ]
    assert (
        any(isinstance(ev, t) for t in expected_evaluator_types)
        for ev in all_benchmarks.evaluators
    )
    assert (
        isinstance(ev_class, AOPC_Comprehensiveness_Evaluation_by_class)
        for ev_class in all_benchmarks.class_based_evaluators
    )


def test_helper_assignment(all_benchmarks):
    for explainer in all_benchmarks.explainers:
        assert explainer.helper == all_benchmarks.helper


def test_helper_override_warning(model_tokenizer_ner):
    model_ner, tokenizer_ner = model_tokenizer_ner
    explainer_with_helper = SHAPExplainer(
        model_ner, tokenizer_ner, helper=SequenceClassificationHelper
    )
    with pytest.warns(UserWarning, match="Overriding helper for explainer"):
        Benchmark(
            model_ner,
            tokenizer_ner,
            task_name="ner",
            explainers=[explainer_with_helper],
        )


# Scoring Checks
def test_scoring_len_and_output(all_benchmarks, cache):
    text = "The weather in London sucks"
    labels = ["weather complaint", "traffic"]
    if all_benchmarks.task_name == "zero-shot-text-classification":
        score = all_benchmarks.score(
            text, options=labels, return_probs=True, return_dict=True
        )
        cache.set("zero-shot-score", score)
        # caching the score of the zero-shot since it will be used later
        expected_labels = labels
    else:
        score = all_benchmarks.score(text, return_dict=True)
        expected_labels = list(all_benchmarks.targets.values())

    if all_benchmarks.task_name == "ner":
        assert all(
            all(label in token_scores[1].keys() for label in expected_labels)
            for token_scores in score.values()
        )
    else:
        assert all(label in score for label in expected_labels)
        assert len(score) == len(expected_labels)
        assert math.isclose(sum(score.values()), 1, abs_tol=0.01)


# Explainer Checks
@pytest.mark.parametrize(
    "explainer",
    [SHAPExplainer, LIMEExplainer, GradientExplainer, IntegratedGradientExplainer],
    indirect=True,
)
@pytest.mark.parametrize(
    "text, target, expected_tokens, expected_target_pos_idx, target_token, task",
    [
        (
            "You look stunning!",
            1,
            ["[CLS]", "you", "look", "stunning", "!", "[SEP]"],
            1,
            None,
            "text-classification",
        ),
        (
            "A tennis game with two females playing.",
            "contradiction",
            ["[CLS]", "▁A", "▁tennis", "▁game", "▁with"],
            2,
            None,
            "nli",
        ),
        (
            "I am John and I live in New York",
            "I-LOC",
            ["[CLS]","I","am","John","and","I","live","in", "New","York","[SEP]",],
            6,
            "York",
            "ner",
        ),
        (
            "The weather in London sucks",
            "entailment",
            ['[CLS]', '▁The', '▁weather', '▁in', '▁London', '▁', 'suck', 's', '[SEP]', '▁This', '▁is', '▁weather', '▁', 'complaint', '[SEP]'],
            0,
            None,
            "zero-shot-text-classification",
        ),
    ],
    ids=["textclass_example", "nli_example", "ner_example", "zero_shot_example"],
)
def test_explainers(
    explainer,
    model_and_tokenizer,
    text,
    target,
    expected_tokens,
    expected_target_pos_idx,
    target_token,
    task,
    cache,
):
    _, _, model_task = model_and_tokenizer
    if model_task != task:
        pytest.skip(f"Skipping {model_task} as it does not match the task {task}")

    if task == "zero-shot-text-classification":
        scores = cache.get("zero-shot-score", {})
        target_option = max(scores, key=scores.get) if scores else None
        sep_token = '[SEP]'
        text = [text + f" {sep_token} " + 'This is {}'.format(target_option)]
    else:
        target_option = None

    explanation = (
        explainer(
            text, target=target, target_token=target_token, target_option=target_option
        )
        if isinstance(explainer, (SHAPExplainer, LIMEExplainer))
        else explainer(text, target=target, target_token=target_token)
    )
    # target_token is for NER and target_option is for zero-shot (remember in SHAP it is ignored)
    assert explanation.tokens[: len(expected_tokens)] == expected_tokens
    assert explanation.target_pos_idx == expected_target_pos_idx
