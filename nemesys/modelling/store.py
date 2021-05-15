from collections import Counter

from typing import Iterable

import torch


def rearrange(
    blocks: Iterable[torch.Tensor],
    groups: Iterable,
    device: torch.device = torch.device("cpu"),
):
    group_to_address = dict()
    split_sizes = Counter()

    last = 0
    for group in groups:
        if group not in group_to_address:
            group_to_address[group] = last
            last += 1

        split_sizes[group_to_address[group]] += 1

    addresses = [group_to_address[group] for group in groups]

    offsets = [0]
    for i in range(len(split_sizes) - 1):
        offsets.append(offsets[-1] + split_sizes[i])

    address_counter = Counter()
    permutations = list()

    for address in addresses:
        permutations.append(offsets[address] + address_counter[address])
        address_counter[address] += 1

    permutations = torch.tensor(
        permutations, dtype=torch.long, device=device, requires_grad=False
    )

    megablock = torch.stack(blocks)
    permuted_megablock = megablock[:, permutations]
    permuted_blocks = permuted_megablock.split(split_sizes)

    return permuted_blocks
