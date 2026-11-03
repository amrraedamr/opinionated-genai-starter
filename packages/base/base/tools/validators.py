import re


def validate_name(name: str) -> str:
    if not re.match(r"^[A-Za-z]+(?:[-'][A-Za-z]+)*$", name):
        raise ValueError(
            "Name must contain only letters, hyphens, or apostrophes, and cannot start or end with a hyphen or apostrophe."
        )
    return name


def validate_password(password: str) -> str:
    violations = []
    if len(password) < 8:
        violations.append("Password must be at least 8 characters long.")
    if not re.search(r"[A-Z]", password):
        violations.append("Password must contain at least one uppercase letter.")
    if not re.search(r"[a-z]", password):
        violations.append("Password must contain at least one lowercase letter.")
    if not re.search(r"\d", password):
        violations.append("Password must contain at least one digit.")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        violations.append("Password must contain at least one special character.")
    if violations:
        raise ValueError("\n".join(violations))
    return password
