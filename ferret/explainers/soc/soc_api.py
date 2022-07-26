from builtins import breakpoint

from ferret.explainers.soc.processor import Processor
from .soc_algo import _SamplingAndOcclusionAlgo
from .lm import BiGRULanguageModel
from .train_lm import do_train_lm
import os, logging, torch, pickle
import json

logger = logging.getLogger(__name__)


def is_lm_trained(lm_dir):
    if not os.path.isdir(lm_dir):
        return False
    files = [f.startswith("best_") for f in os.listdir(lm_dir)]
    if not files:
        return False
    return True


class SamplingAndOcclusionExplain:
    def __init__(
        self,
        model,
        configs,
        tokenizer,
        output_path,
        device,
        lm_dir=None,
        train_dataloader=None,
        dev_dataloader=None,
        vocab=None,
    ):
        self.configs = configs
        self.model = model
        self.lm_dir = lm_dir
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.vocab = vocab
        self.output_path = output_path
        self.device = device
        self.hiex = configs.hiex
        self.tokenizer = tokenizer

        self.lm_model = self.detect_and_load_lm_model()

        self.algo = _SamplingAndOcclusionAlgo(
            model, tokenizer, self.lm_model, output_path, configs
        )

        self.use_padding_variant = configs.use_padding_variant
        try:
            self.output_file = open(self.output_path, "w" if not configs.hiex else "wb")
        except FileNotFoundError:
            self.output_file = None
        self.output_buffer = []

        # for explanation regularization
        self.neutral_words_file = configs.neutral_words_file
        self.neutral_words_ids = None
        self.neutral_words = None
        # self.debug = debug

    def detect_and_load_lm_model(self):
        if not self.lm_dir:
            self.lm_dir = "runs/lm/"
        if not os.path.isdir(self.lm_dir):
            os.mkdir(self.lm_dir)

        file_name = None
        for x in os.listdir(self.lm_dir):
            if x.startswith("best"):
                file_name = x
                break
        if not file_name:
            self.train_lm()
            for x in os.listdir(self.lm_dir):
                if x.startswith("best"):
                    file_name = x
                    break
        lm_model = torch.load(open(os.path.join(self.lm_dir, file_name), "rb"))
        return lm_model

    def set_lm_device(self, device):
        self.lm_model.set_device(device)

    # def train_lm(self):
    #     logger.info("Missing pretrained LM. Now training")
    #     model = BiGRULanguageModel(
    #         self.configs, vocab=self.vocab, device=self.device
    #     ).to(self.device)
    #     do_train_lm(
    #         model,
    #         lm_dir=self.lm_dir,
    #         lm_epochs=20,
    #         train_iter=self.train_dataloader,
    #         dev_iter=self.dev_dataloader,
    #     )

    def word_level_explanation_bert(
        self, input_ids, input_mask, segment_ids, label=None
    ):
        # requires batch size is 1
        # get sequence length
        i = 0
        while i < input_ids.size(1) and input_ids[0, i] != 0:  # pad
            i += 1
        inp_length = i
        # do not explain [CLS] and [SEP]
        spans, scores = [], []
        for i in range(inp_length):  # yes indeed, please explain [CLS] and [SEP]
            span = (i, i)
            spans.append(span)
            if not self.use_padding_variant:
                score = self.algo.do_attribution(
                    input_ids, input_mask, segment_ids, span, label
                )
            else:
                score = self.algo.do_attribution_pad_variant(
                    input_ids, input_mask, segment_ids, span, label
                )
            scores.append(score)
        # inp = input_ids.view(-1).cpu().numpy()
        # s = self.algo.repr_result_region(inp, spans, scores)
        # self.output_file.write(s + '\n')
        return scores

    def hierarchical_explanation_bert(
        self, input_ids, input_mask, segment_ids, label=None
    ):
        tab_info = self.algo.do_hierarchical_explanation(
            input_ids, input_mask, segment_ids, label
        )
        self.output_buffer.append(tab_info)
        # currently store a pkl after explaining each instance
        self.output_file = open(self.output_path, "w" if not self.hiex else "wb")
        pickle.dump(self.output_buffer, self.output_file)
        self.output_file.close()

    def _initialize_neutral_words(self):
        f = open(self.neutral_words_file)
        neutral_words = []
        neutral_words_ids = set()
        for line in f.readlines():
            word = line.strip().split("\t")[0]
            canonical = self.tokenizer.tokenize(word)
            if len(canonical) > 1:
                canonical.sort(key=lambda x: -len(x))
                print(canonical)
            word = canonical[0]
            neutral_words.append(word)
            neutral_words_ids.add(self.tokenizer.vocab[word])
        self.neutral_words = neutral_words
        self.neutral_words_ids = neutral_words_ids
        assert neutral_words

    def compute_explanation_loss(
        self,
        input_ids_batch,
        input_mask_batch,
        segment_ids_batch,
        label_ids_batch,
        do_backprop=False,
    ):
        if self.neutral_words is None:
            self._initialize_neutral_words()
        batch_size = input_ids_batch.size(0)
        neutral_word_scores, cnt = [], 0
        for b in range(batch_size):
            input_ids, input_mask, segment_ids, label_ids = (
                input_ids_batch[b],
                input_mask_batch[b],
                segment_ids_batch[b],
                label_ids_batch[b],
            )
            nw_positions = []
            for i in range(len(input_ids)):
                word_id = input_ids[i].item()
                if word_id in self.neutral_words_ids:
                    nw_positions.append(i)
            # only generate explanations for neutral words
            for i in range(len(input_ids)):
                word_id = input_ids[i].item()
                if word_id in self.neutral_words_ids:
                    x_region = (i, i)
                    # score = self.algo.occlude_input_with_masks_and_run(input_ids, input_mask, segment_ids,
                    #                                                   [x_region], nb_region, label_ids,
                    #                                                    return_variable=True)
                    if not self.configs.use_padding_variant:
                        score = self.algo.do_attribution(
                            input_ids,
                            input_mask,
                            segment_ids,
                            x_region,
                            label_ids,
                            return_variable=True,
                            additional_mask=nw_positions,
                        )
                    else:
                        score = self.algo.do_attribution_pad_variant(
                            input_ids,
                            input_mask,
                            segment_ids,
                            x_region,
                            label_ids,
                            return_variable=True,
                            additional_mask=nw_positions,
                        )
                    score = self.configs.reg_strength * (score ** 2)

                    if do_backprop:
                        score.backward()

                    neutral_word_scores.append(score.item())

        if neutral_word_scores:
            return sum(neutral_word_scores), len(neutral_word_scores)
        else:
            return 0.0, 0
