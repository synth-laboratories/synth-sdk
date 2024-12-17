import hashlib


def get_system_id(system_name: str) -> str:
    """Create a deterministic system_id from system_name using SHA-256."""
    if not system_name:
        raise ValueError("system_name cannot be empty")
    # Create SHA-256 hash of system_name
    hash_object = hashlib.sha256(system_name.encode())
    # Take the first 16 characters of the hex digest for a shorter but still unique ID
    return hash_object.hexdigest()[:16]
