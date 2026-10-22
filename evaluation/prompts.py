"""Canonical original ALBEF classification prompt expansion."""

def build_prompts_for_label(label):
    """
    Multi-prompt template expansion for a single VinDr label.

    Special case:
      - For "No finding" only use the bare label, because
        templates like "There is evidence of No finding" don't make sense.
    """
    clean_label = label.replace("_", " ")

    if clean_label.strip().lower() == "no finding":
        return ["No finding"]

    templates = [
        "{label}",
        "A chest X-ray showing {label}.",
        "Chest radiograph demonstrating {label}.",
        "There is evidence of {label}.",
        "Findings are consistent with {label}.",
    ]
    return [t.format(label=clean_label) for t in templates]
