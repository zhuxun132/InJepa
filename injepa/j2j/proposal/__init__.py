"""Public ProposalJEPA surface."""

from .losses import ProperMixtureTerms, proper_mixture_nll, proper_mixture_terms
from .model import ProposalJEPA, ProposalOutput

__all__ = [
    "ProposalJEPA",
    "ProposalOutput",
    "ProperMixtureTerms",
    "proper_mixture_nll",
    "proper_mixture_terms",
]
