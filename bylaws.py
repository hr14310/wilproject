"""
By-law configuration and definitions.
Extensible list of by-laws that the system monitors for violations.
Add new by-laws by appending to the BYLAWS list.
"""

BYLAWS = [
    {
        "id": "BL-001",
        "name": "Leash Law",
        "description": "Dogs must be on a leash at all times in public spaces.",
        "visual_cues": ["dog without leash", "dog roaming freely", "unleashed dog", "dog off-leash near people"]
    },
    {
        "id": "BL-002",
        "name": "No Trespassing / Fence Jumping",
        "description": "Persons must not climb, jump over, or cross fences into restricted areas.",
        "visual_cues": ["person climbing fence", "person jumping over fence", "person scaling barrier", "person crossing fence line"]
    },
    {
        "id": "BL-003",
        "name": "Littering Prohibition",
        "description": "Persons must not discard waste in public spaces.",
        "visual_cues": ["person throwing trash on ground", "person dropping garbage", "throwing bottle or can", "leaving trash behind"]
    },
    {
        "id": "BL-004",
        "name": "Public Violence / Affray",
        "description": "Physical violence, fighting, or threatening behaviour in public is prohibited.",
        "visual_cues": ["person striking another", "physical altercation between persons", "aggressive threatening gestures", "person attacking another"]
    },
    {
        "id": "BL-005",
        "name": "Robbery",
        "description": "Taking property from a person by force, threat, or intimidation is prohibited.",
        "visual_cues": ["person forcibly taking item from another", "person threatening for valuables", "armed robbery", "person snatching belongings"]
    },
    {
        "id": "BL-006",
        "name": "Vandalism / Property Damage",
        "description": "Persons must not deface, damage, or vandalize public or private property.",
        "visual_cues": ["person spray painting surface", "person vandalizing wall or structure", "person damaging public property", "person writing graffiti"]
    }
]


def get_bylaw_by_id(bylaw_id):
    """Retrieve a by-law by its ID."""
    for bylaw in BYLAWS:
        if bylaw["id"] == bylaw_id:
            return bylaw
    return None


def get_all_bylaws():
    """Return all by-law definitions."""
    return BYLAWS
