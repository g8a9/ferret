import copy


class PertubationHelper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def edit_one_token(self, input_ids, strategy, mask_token_id=None):
        samples = list()

        for occ_idx in range(len(input_ids)):
            sample = copy.copy(input_ids)

            if strategy == "remove":
                sample.pop(occ_idx)
            elif strategy == "mask":
                if mask_token_id is None:
                    mask_token_id = self.tokenizer.mask_token_id
                sample[occ_idx] = mask_token_id
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            samples.append(sample)
        return samples
