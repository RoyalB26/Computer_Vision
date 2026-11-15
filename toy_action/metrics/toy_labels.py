"""Canonical Toy Action benchmark action labels and definitions."""

# 6 Symbols
TOY_SYMBOLS = ["x", "y", "o", "T", "8", "O"]

# 4 Action Directions
TOY_ACTIONS = ["gotop", "gobottom", "goleft", "goright"]

# 12 Symbol Variants (6 single + 6 group '+')
TOY_SYMBOL_VARIANTS = []
for s in TOY_SYMBOLS:
    TOY_SYMBOL_VARIANTS.append(s)
    TOY_SYMBOL_VARIANTS.append(f"{s}+")

# 48 Total Action-Object Classes
TOY_ACTION_LABELS = [f"{s}:{a}" for s in TOY_SYMBOL_VARIANTS for a in TOY_ACTIONS]

# Sub-group splits
SYMBOL_SINGLE_LABELS = [c for c in TOY_ACTION_LABELS if not c.split(":")[0].endswith("+")]
SYMBOL_GROUP_LABELS = [c for c in TOY_ACTION_LABELS if c.split(":")[0].endswith("+")]
