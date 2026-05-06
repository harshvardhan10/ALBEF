import torch


def build_token_mask(input_ids, attention_mask, target_token_ids):
    """
    Build token mask for target phrase, e.g. 'cardiomegaly'.

    Args:
        input_ids:
            Tensor [B, T]

        attention_mask:
            Tensor [B, T]

        target_token_ids:
            list[int]

    Returns:
        token_mask:
            Bool Tensor [B, T]
    """

    B, T = input_ids.shape
    device = input_ids.device

    token_mask = torch.zeros((B, T), dtype=torch.bool, device=device)

    target = torch.tensor(target_token_ids, device=device)
    n = len(target_token_ids)

    for start in range(T - n + 1):
        span = input_ids[:, start:start + n]
        match = (span == target.unsqueeze(0)).all(dim=1)
        valid = attention_mask[:, start:start + n].bool().all(dim=1)

        token_mask[match & valid, start:start + n] = True

    return token_mask