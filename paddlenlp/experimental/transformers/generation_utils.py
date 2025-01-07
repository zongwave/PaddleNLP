# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import os
from typing import List, Union

import paddle
import paddle.nn.functional as F

from paddlenlp.generation import GenerationMixin, LogitsProcessor, LogitsProcessorList

__all__ = ["GenerationInferenceModel", "GenerationBlockInferenceModel", "GenerationAvxInferenceModel"]


def use_faster_top_p_sampling():
    """Get the value of the 'USE_FASTER_TOP_P_SAMPLING' environment variable."""
    return os.getenv("USE_FASTER_TOP_P_SAMPLING", "False") in ["True", "1", "true"]


class ForcedDecodingEOSTokenLogitsProcessor(LogitsProcessor):
    """
    This `LogitsProcessor` enforces the last generated token to be the selected `forced_eos_token`.

    Args:
        max_length (int): The maximum length of the sequence to be generated.
        forced_eos_token_id (int): The id of the token to be generated as the last token.
    """

    def __init__(self, max_decoding_step: int, forced_eos_token_id: Union[int, List[int]]):
        self.max_decoding_step = max_decoding_step
        self.forced_eos_token_id = forced_eos_token_id

    def __call__(self, input_ids, scores, decoding_step):
        if decoding_step == self.max_decoding_step:
            scores[:] = paddle.finfo(scores.dtype).min
            scores[:, self.forced_eos_token_id] = 0
        return scores


def tensors_to_cpu(*tensors):
    return [tensor.cpu() for tensor in tensors]


def tensors_to_device(device, *tensors):
    return [tensor.to(device) for tensor in tensors]


def ref_set_value_by_flags_and_idx(pre_ids_all, pre_ids, step_idx, stop_flags):
    if True:
        import paddlenlp_ops

        device = pre_ids_all.place
        pre_ids_all, pre_ids, step_idx, stop_flags = tensors_to_cpu(pre_ids_all, pre_ids, step_idx, stop_flags)
        stop_flags = paddlenlp_ops.set_value_by_flags_and_idx(pre_ids_all, pre_ids, step_idx, stop_flags)
        stop_flags, pre_ids_all, pre_ids, step_idx, stop_flags = tensors_to_device(
            device, stop_flags, pre_ids_all, pre_ids, step_idx, stop_flags
        )
        return stop_flags
    else:
        dim0, dim1 = pre_ids_all.shape

        pre_ids_all.flatten_()
        step_idx.flatten_()
        stop_flags.flatten_()
        pre_ids.flatten_()

        valid_step_idx = paddle.where(step_idx >= 0, step_idx, 0)
        condition = (step_idx >= 0) & (~stop_flags)
        dst_idx = paddle.arange(0, dim0) * dim1 + valid_step_idx
        dst_idx = dst_idx[condition]
        src_idx = paddle.nonzero(condition)
        selected_elements = paddle.gather(pre_ids, src_idx)
        pre_ids_all.scatter_(dst_idx, selected_elements)

        paddle.reshape_(pre_ids_all, [dim0, dim1])
        step_idx.unsqueeze_(axis=-1)
        stop_flags.unsqueeze_(axis=-1)
        pre_ids.unsqueeze_(axis=-1)

        return stop_flags


def min_length_logits_process(logits, cur_len, min_len, eos_token_id, bs, length, end_length):
    for bi in range(bs):
        if cur_len[bi] < 0:
            continue
        if cur_len[bi] < min_len[bi]:
            for i in range(end_length):
                logits[bi, eos_token_id[i]] = -1e10


def update_repeat_times(pre_ids, cur_len, repeat_times, bs, length_id):
    for bi in range(bs):
        if cur_len[bi] < 0:
            continue

        for i in range(length_id):
            id = pre_ids[bi][i]
            if id < 0:
                break

            repeat_times[bi][id] += 1


def update_value_by_repeat_times(repeat_times, penalty_scores, frequency_score, presence_score, logits, bs, length):
    for bi in range(bs):
        alpha = penalty_scores[bi]
        beta = frequency_score[bi]
        gamma = presence_score[bi]
        for i in range(length):
            times = repeat_times[bi][i]
            if times == 0:
                continue
            logit_now = logits[bi][i]
            logit_now = logit_now < 0 and logit_now * alpha or logit_now / alpha
            logits[bi][i] = logit_now - times * beta - gamma


def ref_get_token_penalty_multi_scores(
    pre_ids, logits, penalty_scores, frequency_scores, presence_scores, cur_len, min_len, eos_token_id
):
    if True:
        device = pre_ids.place
        (
            pre_ids,
            logits,
            penalty_scores,
            frequency_scores,
            presence_scores,
            cur_len,
            min_len,
            eos_token_id,
        ) = tensors_to_cpu(
            pre_ids, logits, penalty_scores, frequency_scores, presence_scores, cur_len, min_len, eos_token_id
        )
        import paddlenlp_ops

        logits_out = paddlenlp_ops.get_token_penalty_multi_scores(
            pre_ids, logits, penalty_scores, frequency_scores, presence_scores, cur_len, min_len, eos_token_id
        )
        (
            logits_out,
            pre_ids,
            logits,
            penalty_scores,
            frequency_scores,
            presence_scores,
            cur_len,
            min_len,
            eos_token_id,
        ) = tensors_to_device(
            device,
            logits_out,
            pre_ids,
            logits,
            penalty_scores,
            frequency_scores,
            presence_scores,
            cur_len,
            min_len,
            eos_token_id,
        )
        return logits_out
    else:
        shape = logits.shape
        repeat_times = paddle.full(shape, 0, dtype="int32")
        bs = shape[0]
        length = shape[1]
        length_id = pre_ids.shape[1]
        logits_out = logits.clone()
        end_length = eos_token_id.shape[0]

        min_length_logits_process(logits_out, cur_len, min_len, eos_token_id, bs, length, end_length)
        update_repeat_times(pre_ids, cur_len, repeat_times, bs, length_id)
        update_value_by_repeat_times(
            repeat_times, penalty_scores, frequency_scores, presence_scores, logits_out, bs, length
        )

        return logits_out


def ref_top_p_sampling(probs, top_p):
    sorted_probs = paddle.sort(probs, descending=True)
    sorted_indices = paddle.argsort(probs, descending=True)
    cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)

    # Remove tokens with cumulative probs above the top_p, But keep at
    # least min_tokens_to_keep tokens
    sorted_indices_to_remove = cumulative_probs > top_p

    # Keep the first token
    sorted_indices_to_remove = paddle.cast(sorted_indices_to_remove, dtype="int64")
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    # Scatter sorted tensors to original indexing
    sorted_indices = sorted_indices + paddle.arange(probs.shape[0]).unsqueeze(-1) * probs.shape[-1]
    condition = paddle.scatter(
        sorted_indices_to_remove.flatten(), sorted_indices.flatten(), sorted_indices_to_remove.flatten()
    )
    condition = paddle.cast(condition, "bool").reshape(probs.shape)
    probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
    if len(probs.shape) > 2:
        probs = paddle.reshape(probs, [probs.shape[0], -1])
    next_tokens = paddle.multinomial(paddle.cast(probs, paddle.float32))
    return next_tokens


def ref_set_stop_value_multi_ends(topk_ids, stop_flags, end_ids):
    import paddlenlp_ops

    place = topk_ids.place
    topk_ids, stop_flags, end_ids = tensors_to_cpu(topk_ids, stop_flags, end_ids)
    result_topk_ids, result_stop_flags = paddlenlp_ops.set_stop_value_multi_ends(topk_ids, stop_flags, end_ids)
    result_topk_ids, result_stop_flags, topk_ids, stop_flags, end_ids = tensors_to_device(
        place, result_topk_ids, result_stop_flags, topk_ids, stop_flags, end_ids
    )
    return result_topk_ids, result_stop_flags


class GenerationInferenceModel(GenerationMixin):
    @classmethod
    def get_cache_kvs_shape(cls, max_batch_size: int = None, max_length: int = None) -> list[list[int]]:
        raise NotImplementedError

    def to_static(self, output_path: str, config: dict):
        dtype = config.get("dtype", paddle.get_default_dtype())

        cache_kvs_shapes = self.get_cache_kvs_shape(self.config, max_length=config.get("max_length", None))
        export_precache = config.get("export_precache", False)
        if export_precache:
            precache_input_spec = [
                paddle.static.InputSpec(shape=[2, None, None, None, None], dtype=dtype, name=f"pre_caches_{i}")
                for i in range(len(cache_kvs_shapes))
            ]
        else:
            precache_input_spec = None

        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),  # input_ids
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype=dtype, name="attention_mask"),  # attention_mask
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="position_ids"),  # position_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="penalty_score"),  # penalty_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="frequency_score"),  # frequency_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="presence_score"),  # presence_score
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="min_length"),  # min_decode_length
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="max_length"),  # max_decode_length
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="temperature"),  # temperature
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="top_p"),  # top_p
            paddle.static.InputSpec(shape=[None], dtype="int64", name="eos_token_id"),  # eos_token_id
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_encoder"),  # seq_len_encoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_decoder"),  # seq_len_decoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="step_idx"),  # step_idx
            paddle.static.InputSpec(shape=[None, 1], dtype="bool", name="stop_flags"),  # stop_flags
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_ids"),  # tgt_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_pos"),  # tgt_pos
            paddle.static.InputSpec(
                shape=[None, 1, 1, None], dtype=dtype, name="tgt_generation_mask"
            ),  # tgt_generation_mask
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="pre_ids"),  # pre_ids
            paddle.static.InputSpec(shape=[1], dtype="int64", name="stop_nums"),  # stop_nums
            [
                paddle.static.InputSpec(
                    shape=shape,
                    dtype=dtype,
                    name="cache_kvs_{}".format(i),
                )
                for i, shape in enumerate(cache_kvs_shapes)
            ],  # cache_kvs
            None,  # inputs_embeds
            config.get("logits_processors", None),
            precache_input_spec,
        ]
        # use "==" to distingusih between chatglm and chatglm_v2.
        if self.config["model_type"] and "chatglm" == self.config.model_type.lower():
            input_spec[2] = paddle.static.InputSpec(
                shape=[None, None, None], dtype="int64", name="position_ids"
            )  # position_ids
            input_spec[16] = paddle.static.InputSpec(shape=[None, 2, 1], dtype="int64", name="tgt_pos")  # tgt_pos
        elif self.config["model_type"] and "gpt" in self.config.model_type:
            input_spec[2] = paddle.static.InputSpec(shape=[None], dtype="int64", name="position_ids")  # position_ids
        model = paddle.jit.to_static(self.generate, input_spec=input_spec)
        paddle.jit.save(
            model, output_path, skip_prune_program=True
        )  # Note(Zhengzekang): If we prune program it may cause some inference error.

    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        seq_len = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
            seq_len = encoder_output.shape[1]
        return paddle.ones([batch_size, seq_len], dtype="int64") * bos_token_id

    @paddle.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        penalty_score=None,
        frequency_score=None,
        presence_score=None,
        min_length=None,
        max_length=None,
        temperature=None,
        top_p=None,
        eos_token_id=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        step_idx=None,
        stop_flags=None,
        tgt_ids=None,
        tgt_pos=None,
        tgt_generation_mask=None,
        pre_ids=None,
        stop_nums=None,
        cache_kvs=[],
        inputs_embeds=None,
        logits_processors=None,
        pre_caches=None,
        **model_kwargs,
    ):
        model_kwargs["position_ids"] = position_ids
        model_kwargs["attention_mask"] = attention_mask

        model_kwargs["seq_len_encoder"] = seq_len_encoder
        model_kwargs["seq_len_decoder"] = seq_len_decoder
        model_kwargs["tgt_ids"] = tgt_ids
        model_kwargs["tgt_generation_mask"] = tgt_generation_mask
        model_kwargs["tgt_pos"] = tgt_pos
        model_kwargs["step_idx"] = step_idx
        model_kwargs["stop_flags"] = stop_flags
        model_kwargs["pre_ids"] = pre_ids
        model_kwargs["min_dec_len"] = min_length
        model_kwargs["max_dec_len"] = max_length
        model_kwargs["stop_nums"] = stop_nums
        model_kwargs["penalty_score"] = penalty_score
        model_kwargs["frequency_score"] = frequency_score
        model_kwargs["presence_score"] = presence_score
        model_kwargs["logits_processors"] = logits_processors or LogitsProcessorList()
        if pre_caches is not None:
            model_kwargs["pre_caches"] = pre_caches

        ret = self.sample(
            input_ids,
            eos_token_id,
            top_p=top_p,
            cache_kvs=cache_kvs,
            temperature=temperature,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

        return ret

    def update_model_kwargs_for_generation(self, cache, just_decoder, next_tokens, eos_token_id, model_kwargs):
        if cache is None:
            model_kwargs["step_idx"] = paddle.where(
                model_kwargs["seq_len_encoder"] == 0,
                model_kwargs["step_idx"],
                model_kwargs["step_idx"] + 1,
            )
        else:
            model_kwargs["step_idx"] = paddle.where(
                model_kwargs["stop_flags"],
                model_kwargs["step_idx"],
                model_kwargs["step_idx"] + 1,
            )
        length_cond = paddle.greater_equal(model_kwargs["step_idx"], model_kwargs["max_dec_len"])
        model_kwargs["stop_flags"] = paddle.logical_or(model_kwargs["stop_flags"], length_cond)
        if cache is None:
            next_tokens = paddle.where(just_decoder, paddle.full_like(next_tokens, -1), next_tokens)
        # from paddlenlp_ops import set_stop_value_multi_ends

        next_tokens, model_kwargs["stop_flags"] = ref_set_stop_value_multi_ends(
            next_tokens, model_kwargs["stop_flags"], eos_token_id
        )  # multi ends

        if cache is None:
            # encoder's generation
            model_kwargs["tgt_ids"] = paddle.where(just_decoder, model_kwargs["tgt_ids"], next_tokens)
            if self.config["position_encoding_2d"] and self.config.position_encoding_2d is True:
                tgt_pos = model_kwargs["tgt_pos"]
                new_position_id = tgt_pos[:, 0, :].clone()
                new_block_id = tgt_pos[:, 1, :].clone()
                new_block_id = new_block_id + 1

                model_kwargs["tgt_pos"] = paddle.concat(
                    [new_position_id.unsqueeze(1), new_block_id.unsqueeze(1)], axis=1
                )
            else:
                model_kwargs["tgt_pos"] = paddle.where(
                    just_decoder, model_kwargs["tgt_pos"], model_kwargs["tgt_pos"] + 1
                )
            """
            model_kwargs["seq_len_decoder"] = paddle.where(
                model_kwargs["stop_flags"],
                model_kwargs["seq_len_decoder"] - model_kwargs["seq_len_decoder"],
                model_kwargs["seq_len_decoder"],
            )
            """
            model_kwargs["seq_len_decoder"] = paddle.where(
                model_kwargs["stop_flags"],
                paddle.zeros_like(model_kwargs["seq_len_decoder"]),
                model_kwargs["seq_len_decoder"],
            )

        else:
            model_kwargs["tgt_ids"] = next_tokens
            if self.config["position_encoding_2d"] and self.config.position_encoding_2d is True:
                tgt_pos = model_kwargs["tgt_pos"]
                new_position_id = tgt_pos[:, 0, :].clone()
                new_block_id = tgt_pos[:, 1, :].clone()
                new_block_id = new_block_id + 1

                model_kwargs["tgt_pos"] = paddle.concat(
                    [new_position_id.unsqueeze(1), new_block_id.unsqueeze(1)], axis=1
                )
            else:
                model_kwargs["tgt_pos"] = paddle.where(
                    model_kwargs["stop_flags"],
                    model_kwargs["tgt_pos"],
                    model_kwargs["tgt_pos"] + 1,
                )

            model_kwargs["seq_len_decoder"] = paddle.where(
                model_kwargs["stop_flags"],
                model_kwargs["seq_len_decoder"],
                model_kwargs["seq_len_decoder"] + 1,
            )

            # a = model_kwargs["seq_len_decoder"]
            # Tensor(shape=[1, 1], dtype=int32, place=Place(intel_hpu:5), stop_gradient=True, [[65]])
            # OSError: synLaunch() failed = 30

            model_kwargs["seq_len_decoder"] = paddle.where(
                model_kwargs["stop_flags"],
                # model_kwargs["seq_len_decoder"] - model_kwargs["seq_len_decoder"],
                paddle.zeros_like(model_kwargs["seq_len_decoder"]),
                model_kwargs["seq_len_decoder"],
            )

        model_kwargs["next_tokens"] = next_tokens
        return model_kwargs

    def sample(
        self,
        input_ids=None,
        eos_token_id=None,
        cache_kvs=[],
        top_p=None,
        temperature=None,
        inputs_embeds=None,
        **model_kwargs,
    ):
        step_idx_ori = paddle.full(shape=[1], dtype="int64", fill_value=1)
        batch_idx = paddle.full(shape=[1], dtype="int32", fill_value=-1)
        model_kwargs["batch_idx"] = batch_idx

        # fake temp next_tokens
        batch = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        next_tokens = paddle.full(shape=[batch, 1], dtype="int32", fill_value=0)

        # let inputs_embeds enter into model_kwargs.
        # because the code below directly use the model_kwargs as a parameter without using inputs_embeds.
        if inputs_embeds is not None:
            model_kwargs["inputs_embeds"] = inputs_embeds
        model_kwargs["all_input_ids"] = input_ids
        logits_processors = model_kwargs.pop("logits_processors")

        def _forward_(**args):
            # cache_kvs is never empty because it is passed as a parameter in def sample.
            model_inputs = self.prepare_inputs_for_generation(input_ids, cache_kvs, **args)
            return self(**model_inputs)

        def _post_process_(outputs, top_p, temperature, step_idx_ori, model_kwargs):
            cache = model_kwargs.get("cache", None)
            just_decoder = model_kwargs["seq_len_encoder"] == 0
            if cache is None:  # first decoder
                step_idx = paddle.where(
                    just_decoder,
                    paddle.full_like(model_kwargs["step_idx"], -1),
                    model_kwargs["step_idx"],
                )  # not update when continue decode
            else:
                step_idx = model_kwargs["step_idx"]
            # from paddlenlp_ops import set_value_by_flags_and_idx

            model_kwargs["stop_flags"] = ref_set_value_by_flags_and_idx(
                model_kwargs["pre_ids"],
                model_kwargs["tgt_ids"],
                step_idx,
                model_kwargs["stop_flags"],
            )
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            logits = paddle.cast(logits, paddle.float32)
            logits = logits_processors(model_kwargs["all_input_ids"], logits, decoding_step=step_idx_ori)

            # from paddlenlp_ops import get_token_penalty_multi_scores

            logits = ref_get_token_penalty_multi_scores(
                model_kwargs["pre_ids"],
                logits,
                model_kwargs["penalty_score"],
                model_kwargs["frequency_score"],
                model_kwargs["presence_score"],
                step_idx,
                model_kwargs["min_dec_len"],
                eos_token_id,
            )
            logits = logits / temperature

            # sample
            probs = F.softmax(logits)

            # compute next_tokens
            if use_faster_top_p_sampling():
                from paddlenlp_ops import top_p_sampling_reject

                next_tokens = top_p_sampling_reject(probs, top_p, 0)
            else:
                device = probs.place
                probs = probs.cpu()
                top_p = top_p.cpu()
                next_tokens = ref_top_p_sampling(probs, top_p)
                probs = probs.to(device)
                top_p = top_p.to(device)
                next_tokens = next_tokens.to(device)

            if self.config.tensor_parallel_degree > 1:
                paddle.distributed.broadcast(next_tokens, 0)

            model_kwargs = self.update_model_kwargs_for_generation(
                cache, just_decoder, next_tokens, eos_token_id, model_kwargs
            )
            next_tokens = model_kwargs["next_tokens"]

            if model_kwargs["all_input_ids"] is None:
                model_kwargs["all_input_ids"] = next_tokens
            else:
                model_kwargs["all_input_ids"] = paddle.concat([model_kwargs["all_input_ids"], next_tokens], axis=1)

            from paddlenlp_ops import save_with_output

            # TODO: remove below two lines, added just for debug
            batch_idx = paddle.full(shape=[1], dtype="int32", fill_value=-1)
            model_kwargs["batch_idx"] = batch_idx

            save_with_output(
                next_tokens,
                model_kwargs["batch_idx"],
                step_idx_ori,
                "real_time_save.temp_ids",
                self.config.tensor_parallel_rank,
            )

            return next_tokens, model_kwargs

        # encoder
        outputs = _forward_(**model_kwargs)
        # first decoder

        next_tokens, model_kwargs = _post_process_(
            outputs,
            top_p,
            temperature,
            step_idx_ori,
            model_kwargs,
        )
        step_idx_ori += 1

        # gives it a value, means we will entered into decoder phase.
        model_kwargs["cache"] = 0

        # decoder
        while paddle.less_than(
            paddle.sum(paddle.cast(model_kwargs["stop_flags"], "int64")),
            model_kwargs["stop_nums"],
        ):
            next_tokens, model_kwargs = _post_process_(
                _forward_(**model_kwargs),
                top_p,
                temperature,
                step_idx_ori,
                model_kwargs,
            )
            step_idx_ori += 1

        return (
            next_tokens,
            model_kwargs["step_idx"],
            paddle.cast(model_kwargs["stop_flags"], "int32"),
            model_kwargs["seq_len_decoder"],
            model_kwargs["tgt_pos"],
        )


class GenerationBlockInferenceModel(GenerationMixin):
    @classmethod
    def get_cache_kvs_shape(cls, max_batch_size: int = None, max_length: int = None) -> list[list[int]]:
        raise NotImplementedError

    def to_static(self, output_path: str, config: dict):
        dtype = config.get("dtype", paddle.get_default_dtype())
        cachekv_dtype = dtype

        cache_kvs_shapes = self.get_cache_kvs_shape(
            self.config, max_batch_size=config.get("max_batch_size", -1), max_length=config.get("max_length", None)
        )
        export_precache = config.get("export_precache", False)
        if export_precache:
            precache_kv_spec = [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype=dtype, name=f"pre_caches_{i}")
                for i in range(len(cache_kvs_shapes))
            ]
        else:
            precache_kv_spec = None
        cachekv_int8_type = config.get("cachekv_int8_type", "None")

        if cachekv_int8_type is not None:
            cachekv_dtype = "uint8"

        if cachekv_int8_type == "dynamic":
            cache_k_quant_scales = [
                paddle.static.InputSpec(
                    shape=[None, self.config.num_attention_heads],
                    dtype="float32",
                    name="k_quant_scales_{}".format(i),
                )
                for i in range(int(len(cache_kvs_shapes) / 2))
            ]

            cache_v_quant_scales = [
                paddle.static.InputSpec(
                    shape=[None, self.config.num_attention_heads],
                    dtype="float32",
                    name="v_quant_scales_{}".format(i),
                )
                for i in range(int(len(cache_kvs_shapes) / 2))
            ]

            cache_k_dequant_scales = [
                paddle.static.InputSpec(
                    shape=[None, self.config.num_attention_heads],
                    dtype="float32",
                    name="k_dequant_scales_{}".format(i),
                )
                for i in range(int(len(cache_kvs_shapes) / 2))
            ]
            cache_v_dequant_scales = [
                paddle.static.InputSpec(
                    shape=[None, self.config.num_attention_heads],
                    dtype="float32",
                    name="v_dequant_scales_{}".format(i),
                )
                for i in range(int(len(cache_kvs_shapes) / 2))
            ]
        else:
            cache_k_quant_scales = None
            cache_v_quant_scales = None
            cache_k_dequant_scales = None
            cache_v_dequant_scales = None

        caches = []
        for i in range(len(cache_kvs_shapes) // 2):
            caches.append(
                paddle.static.InputSpec(
                    shape=cache_kvs_shapes[2 * i], dtype=cachekv_dtype, name="key_caches_{}".format(i)
                )
            )
            caches.append(
                paddle.static.InputSpec(
                    shape=cache_kvs_shapes[2 * i + 1], dtype=cachekv_dtype, name="value_caches_{}".format(i)
                )
            )
        if export_precache:
            src_mask_spec = paddle.static.InputSpec(shape=[None, 1, None, None], dtype=dtype, name="src_mask")
        else:
            src_mask_spec = None

        # bloom model needs src_mask and tgt_mask!
        if "bloom" in self.config.architectures[0].lower():
            src_mask_spec = paddle.static.InputSpec(shape=[None, None, None, None], dtype=dtype, name="src_mask")
            tgt_mask_spec = paddle.static.InputSpec(shape=[None, None, 1, None], dtype=dtype, name="tgt_mask")
        else:
            tgt_mask_spec = None

        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),  # input_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="temperature"),  # temperature
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="top_p"),  # top_p
            paddle.static.InputSpec(shape=[None], dtype="int64", name="eos_token_id"),  # eos_token_id
            src_mask_spec,  # src_mask
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="penalty_score"),  # penalty_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="frequency_score"),  # frequency_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="presence_score"),  # presence_score
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="next_tokens"),  # next_tokens
            paddle.static.InputSpec(shape=[None, 1], dtype="bool", name="is_block_step"),  # is_block_step
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_lens_this_time"),  # seq_lens_this_time
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_lens_encoder"),  # seq_lens_encoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_lens_decoder"),  # seq_lens_decoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="step_idx"),  # step_idx
            paddle.static.InputSpec(shape=[None, 1], dtype="bool", name="stop_flags"),  # stop_flags
            paddle.static.InputSpec(
                shape=[2, None, self.config.max_seq_len, None, None], dtype="float32", name="rope_emb"
            ),  # rope_emb
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="min_length"),  # min_dec_len
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="max_length"),  # max_dec_len
            paddle.static.InputSpec(shape=[1, 1], dtype="int64", name="stop_nums"),  # stop_nums
            paddle.static.InputSpec(shape=[None], dtype="int64", name="bad_tokens"),  # bad_tokens
            paddle.static.InputSpec(shape=[1, 1], dtype="bool", name="not_need_stop"),  # not_need_stop
            paddle.static.InputSpec(shape=[None, None], dtype="int32", name="block_tables"),  # block_tables
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="pre_ids"),  # pre_ids
            precache_kv_spec,
            caches,  # cache_kvs
            cache_k_quant_scales,
            cache_v_quant_scales,
            cache_k_dequant_scales,
            cache_v_dequant_scales,
            tgt_mask_spec,
        ]
        if config.get("speculate_method", None) is not None:
            speculate_spec = [
                paddle.static.InputSpec(shape=[None, None], dtype="int64", name="draft_tokens"),
                paddle.static.InputSpec(shape=[None, None], dtype="int64", name="accept_tokens"),
                paddle.static.InputSpec(shape=[None], dtype="int32", name="accept_num"),
                paddle.static.InputSpec(shape=[None], dtype="int32", name="actual_draft_token_num"),
            ]
            input_spec.extend(speculate_spec)

        model = paddle.jit.to_static(self.generate, input_spec=input_spec)
        paddle.jit.save(
            model, output_path, skip_prune_program=True
        )  # Note(Zhengzekang): If we prune program it may cause some inference error.

    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        seq_len = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
            seq_len = encoder_output.shape[1]
        return paddle.ones([batch_size, seq_len], dtype="int64") * bos_token_id

    def get_output_padding_offset(self, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder):
        """
        In the senerio of speculate decoding, the length of output token after rebuild_padding is no longer bsz.
        So we need to calculate the output_padding_offset after rebuild_padding.
        """
        from paddlenlp_ops import (
            speculate_get_output_padding_offset,
            speculate_get_seq_lens_output,
        )

        seq_lens_output = speculate_get_seq_lens_output(seq_lens_this_time, seq_lens_encoder, seq_lens_decoder)
        out_token_num = paddle.sum(seq_lens_output)
        output_cum_offsets_tmp = paddle.cumsum(self.max_seq_len - seq_lens_output)
        output_padding_offset, output_cum_offsets = speculate_get_output_padding_offset(
            output_cum_offsets_tmp, out_token_num, seq_lens_output, self.max_seq_len
        )
        return output_padding_offset, output_cum_offsets

    @paddle.no_grad()
    def generate(
        self,
        input_ids=None,
        temperature=None,
        top_p=None,
        eos_token_id=None,
        src_mask=None,
        penalty_score=None,
        frequency_score=None,
        presence_score=None,
        next_tokens=None,
        is_block_step=None,
        seq_lens_this_time=None,  # update
        seq_lens_encoder=None,  # update
        seq_lens_decoder=None,  # update
        step_idx=None,
        stop_flags=None,
        rope_emb=None,
        min_length=None,
        max_length=None,
        stop_nums=None,
        bad_tokens=None,
        not_need_stop=None,
        block_tables=None,
        pre_ids=None,
        pre_caches=None,
        cache_kvs=[],
        k_quant_scales=None,
        v_quant_scales=None,
        k_dequant_scales=None,
        v_dequant_scales=None,
        tgt_mask=None,
        draft_tokens=None,
        accept_tokens=None,
        accept_num=None,
        actual_draft_token_num=None,
        **model_kwargs,
    ):

        model_kwargs["input_ids"] = input_ids
        model_kwargs["penalty_score"] = penalty_score
        model_kwargs["frequency_score"] = frequency_score
        model_kwargs["presence_score"] = presence_score
        model_kwargs["seq_lens_this_time"] = seq_lens_this_time
        model_kwargs["seq_lens_encoder"] = seq_lens_encoder
        model_kwargs["seq_lens_decoder"] = seq_lens_decoder
        model_kwargs["step_idx"] = step_idx
        model_kwargs["stop_flags"] = stop_flags
        model_kwargs["min_dec_len"] = min_length
        model_kwargs["max_dec_len"] = max_length
        model_kwargs["stop_nums"] = stop_nums
        model_kwargs["rope_emb"] = rope_emb
        model_kwargs["bad_tokens"] = bad_tokens
        model_kwargs["block_tables"] = block_tables
        model_kwargs["pre_ids"] = pre_ids
        model_kwargs["not_need_stop"] = not_need_stop
        model_kwargs["caches"] = cache_kvs
        model_kwargs["k_quant_scales"] = k_quant_scales
        model_kwargs["v_quant_scales"] = v_quant_scales
        model_kwargs["k_dequant_scales"] = k_dequant_scales
        model_kwargs["v_dequant_scales"] = v_dequant_scales
        model_kwargs["pre_caches"] = pre_caches
        model_kwargs["next_tokens"] = next_tokens
        model_kwargs["is_block_step"] = is_block_step
        model_kwargs["src_mask"] = src_mask
        model_kwargs["tgt_mask"] = tgt_mask
        # speculate decoding related parameters
        model_kwargs["draft_tokens"] = draft_tokens
        model_kwargs["accept_tokens"] = accept_tokens
        model_kwargs["accept_num"] = accept_num
        model_kwargs["actual_draft_token_num"] = actual_draft_token_num

        ret = self.sample(
            eos_token_id,
            top_k=0,
            top_p=top_p,
            temperature=temperature,
            **model_kwargs,
        )
        return ret

    def sample(
        self,
        eos_token_id,
        top_k,
        top_p,
        penalty_score,
        frequency_score,
        presence_score,
        temperature=None,
        min_tokens_to_keep=1,
        **model_kwargs
    ):
        def _forward_(**args):
            model_inputs = self.prepare_inputs_for_generation(**args)
            return self(**model_inputs)

        def _post_process_(
            outputs,
            top_k,
            top_p,
            penalty_score,
            frequency_score,
            presence_score,
            temperature,
            model_kwargs,
        ):
            step_idx = model_kwargs["step_idx"]
            logits = paddle.cast(outputs, paddle.float32)

            # TODO(Wanglongzhi2001): token_penalty of speculative decoding
            if not is_speculative_decoding:
                # from paddlenlp_ops import set_preids_token_penalty_multi_scores

                ref_set_preids_token_penalty_multi_scores(
                    model_kwargs["pre_ids"],
                    model_kwargs["input_ids"],
                    model_kwargs["seq_lens_encoder"],
                    model_kwargs["seq_lens_decoder"],
                    step_idx,
                    model_kwargs["stop_flags"],
                    logits,
                    penalty_score,
                    frequency_score,
                    presence_score,
                    temperature,
                    model_kwargs["bad_tokens"],
                    step_idx,
                    model_kwargs["min_dec_len"],
                    eos_token_id,
                )

            # sample
            probs = F.softmax(logits)

            from paddlenlp_ops import save_output

            # whether speculative decoding
            if not is_speculative_decoding:

                # compute next_tokens
                if use_faster_top_p_sampling():
                    from paddlenlp_ops import top_p_sampling_reject

                    next_tokens = top_p_sampling_reject(probs, top_p, 0)
                else:
                    next_tokens = ref_top_p_sampling(probs, top_p)

                if self.config.tensor_parallel_degree > 1:
                    paddle.distributed.broadcast(next_tokens, 0)

                from paddlenlp_ops import update_inputs_v2

                update_inputs_v2(
                    model_kwargs["stop_flags"],
                    model_kwargs["step_idx"],
                    model_kwargs["not_need_stop"],
                    model_kwargs["seq_lens_this_time"],
                    model_kwargs["seq_lens_encoder"],
                    model_kwargs["seq_lens_decoder"],
                    model_kwargs["max_dec_len"],
                    model_kwargs["input_ids"],
                    model_kwargs["stop_nums"],
                    next_tokens,
                    model_kwargs["is_block_step"],
                    eos_token_id,
                    model_kwargs["next_tokens"],
                )

                save_output(
                    next_tokens,
                    model_kwargs["not_need_stop"],
                    model_kwargs.get("accept_num", None),  # only initialized in speculative decoding
                    self.config.tensor_parallel_rank,
                )
                return next_tokens
            else:
                from paddlenlp_ops import (
                    speculate_set_value_by_flags_and_idx,
                    speculate_verify_and_update,
                    top_p_candidates,
                )

                verify_scores, verify_tokens, actual_candidate_len = top_p_candidates(
                    probs, top_p, model_kwargs["output_padding_offset"], self.max_candidate_len, self.max_seq_len
                )  # [token_num, max_candidate_len]

                # Speculate Verify And Update
                speculate_verify_and_update(
                    model_kwargs["accept_tokens"],
                    model_kwargs["accept_num"],
                    model_kwargs["step_idx"],
                    model_kwargs["seq_lens_encoder"],
                    model_kwargs["seq_lens_decoder"],
                    model_kwargs["stop_flags"],
                    model_kwargs["not_need_stop"],
                    model_kwargs[
                        "draft_tokens"
                    ],  # Both input and output, need to write the last 1 token accepted to position 0.
                    model_kwargs["seq_lens_this_time"],
                    verify_tokens,
                    verify_scores,
                    model_kwargs["max_dec_len"],
                    eos_token_id,
                    model_kwargs["is_block_step"],
                    model_kwargs["output_cum_offsets"],
                    actual_candidate_len,
                    model_kwargs["actual_draft_token_num"],
                    top_p,
                    self.max_seq_len,
                    self.verify_window,
                    True,  # enable_topp
                )

                save_output(
                    model_kwargs["accept_tokens"],
                    model_kwargs["not_need_stop"],
                    model_kwargs["accept_num"],
                    self.config.tensor_parallel_rank,
                )

                # If seq_lens_decoder is 0 (means stop), accept_num should be set to 0
                model_kwargs["accept_num"][model_kwargs["seq_lens_decoder"] == 0] = 0

                # Update pre_ids through accept tokens
                speculate_set_value_by_flags_and_idx(
                    model_kwargs["pre_ids"],
                    model_kwargs["accept_tokens"],
                    model_kwargs["accept_num"],
                    model_kwargs["stop_flags"],
                    model_kwargs["seq_lens_this_time"],
                    model_kwargs["seq_lens_encoder"],
                    model_kwargs["seq_lens_decoder"],
                    model_kwargs["step_idx"],
                )

        is_speculative_decoding = model_kwargs.get("draft_tokens", None) is not None
        if is_speculative_decoding:
            # Prepare output padding offset
            output_padding_offset, output_cum_offsets = self.get_output_padding_offset(
                model_kwargs["seq_lens_this_time"], model_kwargs["seq_lens_encoder"], model_kwargs["seq_lens_decoder"]
            )
            model_kwargs["output_padding_offset"] = output_padding_offset
            model_kwargs["output_cum_offsets"] = output_cum_offsets

        # encoder
        outputs = _forward_(**model_kwargs)  # [bs, 1, dim_embed]
        # first decoder
        next_tokens = _post_process_(
            outputs,
            top_k,
            top_p,
            penalty_score,
            frequency_score,
            presence_score,
            temperature,
            model_kwargs,
        )

        return next_tokens


class GenerationAvxInferenceModel(GenerationMixin):
    @classmethod
    def get_cache_kvs_shape(cls, max_batch_size: int = None, max_length: int = None) -> list[list[int]]:
        raise NotImplementedError

    def to_static(self, output_path: str, config: dict):
        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),  # input_ids
            None,  # attention_mask
            None,  # position_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="penalty_score"),  # penalty_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="frequency_score"),  # frequency_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="presence_score"),  # presence_score
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="min_length"),  # min_decode_length
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="max_length"),  # max_decode_length
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="temperature"),  # temperature
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="top_p"),  # top_p
            paddle.static.InputSpec(shape=[None], dtype="int64", name="eos_token_id"),  # eos_token_id
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_encoder"),  # seq_len_encoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_decoder"),  # seq_len_decoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="step_idx"),  # step_idx
            paddle.static.InputSpec(shape=[None, 1], dtype="bool", name="stop_flags"),  # stop_flags
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_ids"),  # tgt_ids
            None,  # tgt_pos
            None,  # tgt_generation_mask
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="pre_ids"),  # pre_ids
            paddle.static.InputSpec(shape=[1], dtype="int64", name="stop_nums"),  # stop_nums
            None,  # cache_kvs
            None,  # inputs_embeds
            config.get("logits_processors", None),
            None,
        ]
        model = paddle.jit.to_static(self.generate, input_spec=input_spec)
        paddle.jit.save(
            model, output_path, skip_prune_program=True
        )  # Note(Zhengzekang): If we prune program it may cause some inference error.

    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        seq_len = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
            seq_len = encoder_output.shape[1]
        return paddle.ones([batch_size, seq_len], dtype="int64") * bos_token_id

    @paddle.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        penalty_score=None,
        frequency_score=None,
        presence_score=None,
        min_length=None,
        max_length=None,
        temperature=None,
        top_p=None,
        eos_token_id=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        step_idx=None,
        stop_flags=None,
        tgt_ids=None,
        tgt_pos=None,
        tgt_generation_mask=None,
        pre_ids=None,
        stop_nums=None,
        cache_kvs=[],
        inputs_embeds=None,
        logits_processors=None,
        pre_caches=None,
        **model_kwargs,
    ):
        model_kwargs["seq_len_encoder"] = seq_len_encoder
        model_kwargs["seq_len_decoder"] = seq_len_decoder
        model_kwargs["tgt_ids"] = tgt_ids
        model_kwargs["step_idx"] = step_idx
        model_kwargs["stop_flags"] = stop_flags
        model_kwargs["pre_ids"] = pre_ids
        model_kwargs["min_dec_len"] = min_length
        model_kwargs["max_dec_len"] = max_length
        model_kwargs["stop_nums"] = stop_nums
        model_kwargs["penalty_score"] = penalty_score
        model_kwargs["frequency_score"] = frequency_score
        model_kwargs["presence_score"] = presence_score
        model_kwargs["logits_processors"] = logits_processors or LogitsProcessorList()

        ret = self.sample(
            input_ids,
            eos_token_id,
            top_p=top_p,
            cache_kvs=cache_kvs,
            temperature=temperature,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
        return ret

    def update_model_kwargs_for_generation(self, cache, just_decoder, next_tokens, eos_token_id, model_kwargs):
        if cache is None:
            # llama step_idx ++
            model_kwargs["step_idx"] = paddle.where(
                model_kwargs["seq_len_encoder"] == 0,
                model_kwargs["step_idx"],
                model_kwargs["step_idx"] + 1,
            )
        else:
            model_kwargs["step_idx"] = paddle.where(
                model_kwargs["stop_flags"],
                model_kwargs["step_idx"],
                model_kwargs["step_idx"] + 1,
            )

        length_cond = paddle.greater_equal(model_kwargs["step_idx"], model_kwargs["max_dec_len"])
        model_kwargs["stop_flags"] = paddle.logical_or(model_kwargs["stop_flags"], length_cond)
        if cache is None:
            next_tokens = paddle.where(just_decoder, paddle.full_like(next_tokens, -1), next_tokens)
        # from paddlenlp_ops import set_stop_value_multi_ends

        next_tokens, model_kwargs["stop_flags"] = ref_set_stop_value_multi_ends(
            next_tokens, model_kwargs["stop_flags"], eos_token_id
        )  # multi ends

        if cache is None:
            # encoder's generation
            model_kwargs["tgt_ids"] = paddle.where(just_decoder, model_kwargs["tgt_ids"], next_tokens)
            model_kwargs["seq_len_decoder"] = paddle.where(
                model_kwargs["stop_flags"],
                model_kwargs["seq_len_decoder"] - model_kwargs["seq_len_decoder"],
                model_kwargs["seq_len_decoder"],
            )
        else:
            model_kwargs["tgt_ids"] = next_tokens
            model_kwargs["seq_len_decoder"] = paddle.where(
                model_kwargs["stop_flags"],
                model_kwargs["seq_len_decoder"],
                model_kwargs["seq_len_decoder"] + 1,
            )

            model_kwargs["seq_len_decoder"] = paddle.where(
                model_kwargs["stop_flags"],
                model_kwargs["seq_len_decoder"] - model_kwargs["seq_len_decoder"],
                model_kwargs["seq_len_decoder"],
            )

        model_kwargs["next_tokens"] = next_tokens
        return model_kwargs

    def sample(
        self,
        input_ids=None,
        eos_token_id=None,
        cache_kvs=[],
        top_p=None,
        temperature=None,
        inputs_embeds=None,
        **model_kwargs,
    ):
        step_idx_ori = paddle.full(shape=[1], dtype="int64", fill_value=1)
        batch_idx = paddle.full(shape=[1], dtype="int32", fill_value=-1)

        # fake temp next_tokens
        batch = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        next_tokens = paddle.full(shape=[batch, 1], dtype="int32", fill_value=0)

        # let inputs_embeds enter into model_kwargs.
        # because the code below directly use the model_kwargs as a parameter without using inputs_embeds.
        model_kwargs["inputs_embeds"] = inputs_embeds
        model_kwargs["all_input_ids"] = input_ids
        logits_processors = model_kwargs.pop("logits_processors")

        def _forward_(**args):
            # cache_kvs is never empty because it is passed as a parameter in def sample.
            model_inputs = self.prepare_inputs_for_generation(input_ids, **args)
            return self(**model_inputs)

        def _post_process_(outputs, top_p, temperature, step_idx_ori, model_kwargs):
            cache = model_kwargs.get("cache", None)
            just_decoder = model_kwargs["seq_len_encoder"] == 0
            if cache is None:  # first decoder
                step_idx = paddle.where(
                    just_decoder,
                    paddle.full_like(model_kwargs["step_idx"], -1),
                    model_kwargs["step_idx"],
                )  # not update when continue decode
            else:
                step_idx = model_kwargs["step_idx"]

            # from paddlenlp_ops import set_value_by_flags_and_idx

            model_kwargs["stop_flags"] = ref_set_value_by_flags_and_idx(
                model_kwargs["pre_ids"],
                model_kwargs["tgt_ids"],
                step_idx,
                model_kwargs["stop_flags"],
            )
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = paddle.cast(logits, paddle.float32)
            logits = logits_processors(model_kwargs["all_input_ids"], logits, decoding_step=step_idx_ori)

            # from paddlenlp_ops import get_token_penalty_multi_scores

            logits = get_token_penalty_multi_scores(
                model_kwargs["pre_ids"],
                logits,
                model_kwargs["penalty_score"],
                model_kwargs["frequency_score"],
                model_kwargs["presence_score"],
                step_idx,
                model_kwargs["min_dec_len"],
                eos_token_id,
            )
            logits = logits / temperature
            probs = F.softmax(logits)

            from paddlenlp_ops import xft_greedy_search

            next_tokens = xft_greedy_search(probs)

            model_kwargs = self.update_model_kwargs_for_generation(
                cache, just_decoder, next_tokens, eos_token_id, model_kwargs
            )
            next_tokens = model_kwargs["next_tokens"]

            if model_kwargs["all_input_ids"] is None:
                model_kwargs["all_input_ids"] = next_tokens
            else:
                model_kwargs["all_input_ids"] = paddle.concat([model_kwargs["all_input_ids"], next_tokens], axis=1)

            from paddlenlp_ops import save_with_output

            save_with_output(
                next_tokens,
                model_kwargs["batch_idx"],
                step_idx_ori,
                "real_time_save.temp_ids",
                self.config.tensor_parallel_rank,
            )

            return next_tokens, model_kwargs

        # encoder
        outputs = _forward_(**model_kwargs)
        # first decoder
        next_tokens, model_kwargs = _post_process_(
            outputs,
            top_p,
            temperature,
            step_idx_ori,
            model_kwargs,
        )
        step_idx_ori += 1

        # gives it a value, means we will entered into decoder phase.
        model_kwargs["cache"] = 0

        while paddle.less_than(
            paddle.sum(paddle.cast(model_kwargs["stop_flags"], "int64")),
            model_kwargs["stop_nums"],
        ):
            next_tokens, model_kwargs = _post_process_(
                _forward_(**model_kwargs),
                top_p,
                temperature,
                step_idx_ori,
                model_kwargs,
            )
            step_idx_ori += 1
        return (
            next_tokens,
            model_kwargs["step_idx"],
            paddle.cast(model_kwargs["stop_flags"], "int32"),
            model_kwargs["seq_len_decoder"],
            None,
        )
