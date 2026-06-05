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
    "description": "Physical violence or fighting actively occurring in public. Requires visible striking, punching, kicking, or physical contact in aggression. Does NOT include people standing, walking, or gathering near each other.",
    "visual_cues": [
        "person actively punching or striking another person",
        "two or more people physically fighting with contact",
        "person kicking another person aggressively",
        "person grabbing and throwing another person",
        "violent brawl with multiple people hitting each other",
        "person being pinned down aggressively by another"
    ]
    },
    {
    "id": "BL-005",
    "name": "Fire Hazard",
    "description": "Uncontrolled open fire or large flames spreading in a public or indoor space, not including controlled cooking on a stove or barbecue.",
    "visual_cues": [
        "large uncontrolled fire spreading indoors",
        "building or structure on fire with visible flames",
        "open fire burning outdoors in restricted area",
        "fire spreading across floor or walls",
        "heavy smoke filling a room from fire",
        "flames engulfing objects or furniture"]
    },
    {
    "id": "BL-006",
    "name": "Worker Fall",
    "description": "A worker, delivery person, or individual has fallen or collapsed due to an accident, slip, or injury in any public or work setting including streets, doorsteps, or buildings.",
    "visual_cues": [
        "person fallen to the ground",
        "worker lying on ground after falling",
        "delivery person fallen on doorstep or footpath",
        "person collapsed on floor after slipping",
        "worker fallen from height or ladder",
        "person lying on ground unable to get up after fall",
        "person fallen while carrying packages or equipment",
        "worker motionless on floor after accident"]
    },
    {
    "id": "BL-007",
    "name": "Gym Accident",
    "description": "An accident or injury in a gym where a person is visibly in distress, lying motionless, or has fallen uncontrollably from equipment. Does NOT include normal exercise such as pushups, situps, stretching, or planks on the floor.",
    "visual_cues": [
        "person lying motionless on gym floor after falling",
        "person clutching injury after dropping weights",
        "person fallen and not moving near gym equipment",
        "person unconscious or unresponsive on gym floor",
        "barbell or weight fallen on top of person",
        "person falling uncontrollably off treadmill or machine"]
    },
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
