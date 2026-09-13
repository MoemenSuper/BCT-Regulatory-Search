"""Small, evidence-backed regulatory relationship graph for the BCT runtime.

The package deliberately separates automatic *candidate extraction* from the
verified graph consumed by the chatbot.  LLM-produced legal-change edges are
never treated as verified merely because the builder emitted them.
"""

from .models import (
    InstrumentRef,
    RelationshipCandidate,
    RelationshipType,
    VerificationStatus,
)

__all__ = [
    "InstrumentRef",
    "RelationshipCandidate",
    "RelationshipType",
    "VerificationStatus",
]
