"""Value construction for Nexus-owned server 1.0 expression builders."""


def literal(value):
    """Preserve application data when an expression could interpret it as syntax."""
    if isinstance(value, str) and value.startswith("$"):
        return "$" + value
    if isinstance(value, (list, dict)):
        return {"json": value}
    return value
