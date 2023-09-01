# modified functions from https://github.com/vimalabs/VIMA/blob/main/scripts/example.py

import torch
import cv2
import os
import numpy as np
from einops import rearrange
from vima.utils import *
from tokenizers import Tokenizer
from tokenizers import AddedToken
from segmentation.segmentation import InvSegmentater

views = ["pov"]
segmentator = InvSegmentater()
segmentator.load_state_dict(torch.load("segmentation/checkpoint_epoch10.ckpt", map_location="cpu"))


def init_tokenizer():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    _kwargs = {
        "single_word": True,
        "lstrip": False,
        "rstrip": False,
        "normalized": True,
    }

    PLACEHOLDER_TOKENS = [
        AddedToken("{base_obj}", **_kwargs),
        AddedToken("{base_obj_1}", **_kwargs),
        AddedToken("{base_obj_2}", **_kwargs),
        AddedToken("{dragged_obj}", **_kwargs),
        AddedToken("{dragged_obj_1}", **_kwargs),
        AddedToken("{dragged_obj_2}", **_kwargs),
        AddedToken("{dragged_obj_3}", **_kwargs),
        AddedToken("{dragged_obj_4}", **_kwargs),
        AddedToken("{dragged_obj_5}", **_kwargs)
    ]
    PLACEHOLDERS = [token.content for token in PLACEHOLDER_TOKENS]
    tokenizer = Tokenizer.from_pretrained("t5-base")
    tokenizer.add_tokens(PLACEHOLDER_TOKENS)
    return {"tokenizer": tokenizer, "placeholders": PLACEHOLDERS}


def init_cache():
    return {
        "obs_tokens": [],
        "obs_masks": [],
        "action_tokens": []
    }


def prompt_to_forwarddata(policy, prompt_data, tokenizer_data, device):
    prompt_token_type, word_batch, image_batch = prepare_prompt(
        **prompt_data, views=views, **tokenizer_data
    )
    word_batch = word_batch.to(device)
    image_batch = image_batch.to_torch_tensor(device=device)
    prompt_token, prompt_mask = policy.forward_prompt_assembly((prompt_token_type, word_batch, image_batch))
    return {"prompt_token": prompt_token, "prompt_token_mask": prompt_mask}


def obs_to_forwarddata(policy, obs, inference_cache, device):
    obs = prepare_obs(obs, views, device)
    obs = obs.to_torch_tensor(device=device)
    obs_token_this_step, obs_mask_this_step = policy.forward_obs_token(obs)
    obs_token_this_step = obs_token_this_step.squeeze(0)
    obs_mask_this_step = obs_mask_this_step.squeeze(0)
    # adding to cache
    inference_cache["obs_tokens"].append(obs_token_this_step[0])
    inference_cache["obs_masks"].append(obs_mask_this_step[0])
    max_objs = max(x.shape[0] for x in inference_cache["obs_tokens"])
    # creating obs_tokens and masks from cache with padding to forward
    obs_tokens_to_forward, obs_masks_to_forward = [], []
    obs_tokens_this_env, obs_masks_this_env = [], []
    for idx in range(len(inference_cache["obs_tokens"])):
        obs_this_env_this_step = inference_cache["obs_tokens"][idx]
        obs_mask_this_env_this_step = inference_cache["obs_masks"][idx]
        required_pad = max_objs - obs_this_env_this_step.shape[0]
        obs_tokens_this_env.append(
            any_concat(
                [
                    obs_this_env_this_step,
                    torch.zeros(
                        required_pad,
                        obs_this_env_this_step.shape[1],
                        device=device,
                        dtype=obs_this_env_this_step.dtype,
                    ),
                ],
                dim=0,
            )
        )
        obs_masks_this_env.append(
            any_concat(
                [
                    obs_mask_this_env_this_step,
                    torch.zeros(
                        required_pad,
                        device=device,
                        dtype=obs_mask_this_env_this_step.dtype,
                    ),
                ],
                dim=0,
            )
        )
    obs_tokens_to_forward.append(any_stack(obs_tokens_this_env, dim=0))
    obs_masks_to_forward.append(any_stack(obs_masks_this_env, dim=0))
    obs_tokens_to_forward = any_stack(obs_tokens_to_forward, dim=0)
    obs_masks_to_forward = any_stack(obs_masks_to_forward, dim=0)
    obs_tokens_to_forward = obs_tokens_to_forward.transpose(0, 1)
    obs_masks_to_forward = obs_masks_to_forward.transpose(0, 1)

    # creating action_tokens from cache to forward
    action_tokens_to_forward = any_stack([any_stack(inference_cache["action_tokens"], dim=0)], dim=0).transpose(0, 1) if inference_cache["action_tokens"] else None
    return {
        "obs_token": obs_tokens_to_forward,
        "obs_mask": obs_masks_to_forward,
        "action_token": action_tokens_to_forward
    }, inference_cache


def prepare_prompt(prompt, prompt_assets, views, tokenizer, placeholders):
    views = sorted(views)
    encoding = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_ids, prompt_tokens = encoding.ids, encoding.tokens
    assert set(prompt_assets.keys()) == set(
        [token[1:-1] for token in prompt_tokens if token in placeholders]
    )
    filled_prompt = []
    for id, token in zip(prompt_ids, prompt_tokens):
        if token not in placeholders:
            assert "{" not in token and "}" not in token
            filled_prompt.append(id)
        else:
            assert token.startswith("{") and token.endswith("}")
            asset_name = token[1:-1]
            assert asset_name in prompt_assets, f"missing prompt asset {asset_name}"
            asset = prompt_assets[asset_name]
            objects = [asset["segm"]["obj_info"]["obj_id"]]
            obj_repr = {
                "cropped_img": {view: [] for view in views},
                "bbox": {view: [] for view in views},
            }
            for view in views:
                rgb_this_view = asset["rgb"][view]
                segm_this_view = asset["segm"][view]
                bboxes = []
                cropped_imgs = []
                for obj_id in objects:
                    ys, xs = np.nonzero(segm_this_view == obj_id)
                    if len(xs) < 2 or len(ys) < 2:
                        continue
                    xmin, xmax = np.min(xs), np.max(xs)
                    ymin, ymax = np.min(ys), np.max(ys)
                    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
                    h, w = ymax - ymin, xmax - xmin
                    bboxes.append([int(x_center), int(y_center), int(h), int(w)])
                    cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
                    if cropped_img.shape[1] != cropped_img.shape[2]:
                        diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                        pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                        if cropped_img.shape[1] > cropped_img.shape[2]:
                            pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                        else:
                            pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                        cropped_img = np.pad(
                            cropped_img,
                            pad_width,
                            mode="constant",
                            constant_values=0,
                        )
                        assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
                    cropped_img = rearrange(cropped_img, "c h w -> h w c")
                    cropped_img = np.asarray(cropped_img)
                    cropped_img = cv2.resize(
                        cropped_img,
                        (32, 32),
                        interpolation=cv2.INTER_AREA,
                    )
                    cropped_img = rearrange(cropped_img, "h w c -> c h w")
                    cropped_imgs.append(cropped_img)
                bboxes = np.asarray(bboxes)
                cropped_imgs = np.asarray(cropped_imgs)
                obj_repr["bbox"][view] = bboxes
                obj_repr["cropped_img"][view] = cropped_imgs
            filled_prompt.append(obj_repr)
    raw_prompt = [filled_prompt]
    max_n_objs_prompt = {view: 0 for view in views}
    for prompt in raw_prompt:
        for token in prompt:
            if isinstance(token, dict):
                for view in views:
                    max_n_objs_prompt[view] = max(
                        max_n_objs_prompt[view], len(token["cropped_img"][view])
                    )
    raw_prompt_token_type, word_batch, image_batch = [], [], []
    for prompt in raw_prompt:
        token_type = []
        for token in prompt:
            if isinstance(token, int):
                token_type.append(0)
                word_batch.append(token)
            elif isinstance(token, dict):
                token_type.append(1)
                n_objs_prompt = {
                    view: len(token["cropped_img"][view]) for view in views
                }
                # add mask
                token["mask"] = {
                    view: np.ones((n_objs_prompt[view],), dtype=bool)
                    for view in views
                }
                n_objs_to_pad = {
                    view: max_n_objs_prompt[view] - n_objs_prompt[view]
                    for view in views
                }
                objs_pad = {
                    "bbox": {
                        view: np.zeros((n_objs_to_pad[view], 4), dtype=np.int64)
                        for view in views
                    },
                    "cropped_img": {
                        view: np.zeros(
                            (n_objs_to_pad[view], 3, 32, 32),
                            dtype=np.uint8,
                        )
                        for view in views
                    },
                    "mask": {
                        view: np.zeros((n_objs_to_pad[view]), dtype=bool)
                        for view in views
                    },
                }
                token = any_concat([token, objs_pad], dim=0)
                image_batch.append(token)
        raw_prompt_token_type.append(token_type)
    assert sum([len(prompt) for prompt in raw_prompt_token_type]) == len(
        word_batch
    ) + len(image_batch)
    word_batch = any_stack(word_batch, dim=0)
    image_batch = any_to_datadict(stack_sequence_fields(image_batch))

    word_batch = any_to_torch_tensor(word_batch)
    image_batch = image_batch.to_torch_tensor()
    return raw_prompt_token_type, word_batch, image_batch


    assert not (rgb_dict is not None and "rgb" in obs)
    rgb_dict = rgb_dict or obs.pop("rgb")
    segm_dict = obs.pop("segm")
    views = sorted(rgb_dict.keys())
    assert meta["n_objects"] == len(meta["obj_id_to_info"])
    objects = list(meta["obj_id_to_info"].keys())

    L_obs = get_batch_size(obs)

    obs_list = {
        "ee": obs["ee"],
        "objects": {
            "cropped_img": {view: [] for view in views},
            "bbox": {view: [] for view in views},
            "mask": {view: [] for view in views},
        },
    }

    for l in range(L_obs):
        rgb_dict_this_step = any_slice(rgb_dict, np.s_[l])
        segm_dict_this_step = any_slice(segm_dict, np.s_[l])
        for view in views:
            rgb_this_view = rgb_dict_this_step[view]
            segm_this_view = segm_dict_this_step[view]
            bboxes = []
            cropped_imgs = []
            n_pad = 0
            for obj_id in objects:
                ys, xs = np.nonzero(segm_this_view == obj_id)
                if len(xs) < 2 or len(ys) < 2:
                    n_pad += 1
                    continue
                xmin, xmax = np.min(xs), np.max(xs)
                ymin, ymax = np.min(ys), np.max(ys)
                x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
                h, w = ymax - ymin, xmax - xmin
                bboxes.append([int(x_center), int(y_center), int(h), int(w)])
                cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
                if cropped_img.shape[1] != cropped_img.shape[2]:
                    diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                    pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                    if cropped_img.shape[1] > cropped_img.shape[2]:
                        pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                    else:
                        pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                    cropped_img = np.pad(
                        cropped_img, pad_width, mode="constant", constant_values=0
                    )
                    assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
                cropped_img = rearrange(cropped_img, "c h w -> h w c")
                cropped_img = np.asarray(cropped_img)
                cropped_img = cv2.resize(
                    cropped_img,
                    (32, 32),
                    interpolation=cv2.INTER_AREA,
                )
                cropped_img = rearrange(cropped_img, "h w c -> c h w")
                cropped_imgs.append(cropped_img)
            bboxes = np.asarray(bboxes)
            cropped_imgs = np.asarray(cropped_imgs)
            mask = np.ones(len(bboxes), dtype=bool)
            if n_pad > 0:
                bboxes = np.concatenate(
                    [bboxes, np.zeros((n_pad, 4), dtype=bboxes.dtype)], axis=0
                )
                cropped_imgs = np.concatenate(
                    [
                        cropped_imgs,
                        np.zeros(
                            (n_pad, 3, 32, 32),
                            dtype=cropped_imgs.dtype,
                        ),
                    ],
                    axis=0,
                )
                mask = np.concatenate([mask, np.zeros(n_pad, dtype=bool)], axis=0)
            obs_list["objects"]["bbox"][view].append(bboxes)
            obs_list["objects"]["cropped_img"][view].append(cropped_imgs)
            obs_list["objects"]["mask"][view].append(mask)
    for view in views:
        obs_list["objects"]["bbox"][view] = np.stack(
            obs_list["objects"]["bbox"][view], axis=0
        )
        obs_list["objects"]["cropped_img"][view] = np.stack(
            obs_list["objects"]["cropped_img"][view], axis=0
        )
        obs_list["objects"]["mask"][view] = np.stack(
            obs_list["objects"]["mask"][view], axis=0
        )

    obs = any_to_datadict(any_stack([obs_list], dim=0))
    obs = obs.to_torch_tensor()
    obs = any_transpose_first_two_axes(obs)
    return obs


def prepare_obs(obs, views, device):
    obs_list = {
        "objects": {
            "cropped_img": {view: [] for view in views},
            "bbox": {view: [] for view in views},
            "mask": {view: [] for view in views},
        },
    }

    for view in views:
        cropped_imgs = []
        n_pad = 0
        bboxes = image_to_bboxes(obs[view], device)

        for center_x, center_y, h, w in bboxes:
            cropped_img = np.array(obs[view][center_y-h//2:center_y+h//2, center_x-w//2:center_x+w//2])
            cropped_img = rearrange(cropped_img, "h w c -> c h w")

            # adjust width and height of cropped image to same size
            if cropped_img.shape[1] != cropped_img.shape[2]:
                diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                if cropped_img.shape[1] > cropped_img.shape[2]:
                    pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                else:
                    pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                cropped_img = np.pad(
                    cropped_img, pad_width, mode="constant", constant_values=0
                )
                assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
            cropped_img = rearrange(cropped_img, "c h w -> h w c")
            cropped_img = np.asarray(cropped_img)
            cropped_img = cv2.resize(
                cropped_img,
                (32, 32),
                interpolation=cv2.INTER_AREA,
            )
            cropped_img = rearrange(cropped_img, "h w c -> c h w")
            cropped_imgs.append(cropped_img)
        bboxes = np.asarray(bboxes)
        cropped_imgs = np.asarray(cropped_imgs)
        mask = np.ones(len(bboxes), dtype=bool)
        if n_pad > 0:
            bboxes = np.concatenate(
                [bboxes, np.zeros((n_pad, 4), dtype=bboxes.dtype)], axis=0
            )
            cropped_imgs = np.concatenate(
                [
                    cropped_imgs,
                    np.zeros(
                        (n_pad, 3, 32, 32),
                        dtype=cropped_imgs.dtype,
                    ),
                ],
                axis=0,
            )
            mask = np.concatenate([mask, np.zeros(n_pad, dtype=bool)], axis=0)
        obs_list["objects"]["bbox"][view].append(bboxes)
        obs_list["objects"]["cropped_img"][view].append(cropped_imgs)
        obs_list["objects"]["mask"][view].append(mask)
    
    for view in views:
        obs_list["objects"]["bbox"][view] = np.stack(obs_list["objects"]["bbox"][view], axis=0)
        obs_list["objects"]["cropped_img"][view] = np.stack(obs_list["objects"]["cropped_img"][view], axis=0)
        obs_list["objects"]["mask"][view] = np.stack(obs_list["objects"]["mask"][view], axis=0)

    obs = any_to_datadict(any_stack([obs_list], dim=0))
    obs = obs.to_torch_tensor()
    obs["ee"] = torch.tensor([[0]])
    obs = any_transpose_first_two_axes(obs)
    return obs


def image_to_bboxes(img, device):
    global segmentator
    segmentator = segmentator.to(device)
    img = img.moveaxis(-1, 0).to(device).unsqueeze(0).type(torch.float32)
    with torch.no_grad():
        mask = segmentator(img)
        mask = torch.round(torch.sigmoid(mask.squeeze())).numpy().astype(np.uint8)
    
    n, labels = cv2.connectedComponents(mask)
    bboxes = [[float("inf"), float("inf"), -float("inf"), -float("inf")] for _ in range(n)]
    
    for y in range(len(labels)):
        for x in range(len(labels[y])):
            label = labels[y][x]
            if mask[y][x] == 0:
                continue
            bboxes[label][0] = min(bboxes[label][0], x)
            bboxes[label][1] = min(bboxes[label][1], y)
            bboxes[label][2] = max(bboxes[label][2], x)
            bboxes[label][3] = max(bboxes[label][3], y)
    bboxes = list(filter(lambda x: x != [float("inf"), float("inf"), -float("inf"), -float("inf")], bboxes))
    bboxes = np.array(bboxes)
    # (x1, y1, x2, y2) -> (x_center, y_center, h, w)
    x_min = bboxes[:, 0]
    y_min = bboxes[:, 1]
    x_max = bboxes[:, 2]
    y_max = bboxes[:, 3]
    w = x_max - x_min
    h = y_max - y_min
    bboxes = np.stack([x_min+w//2, y_min+h//2, h, w], axis=1)

    bboxes = bboxes[np.all(bboxes[:, 2::] > 5, axis=1)] # sort out small clusters
    return bboxes