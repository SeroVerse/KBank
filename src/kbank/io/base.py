import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto

from kbank.glycan import Glycan
from kbank.sugars import Sugars


# Errors and warnings --------------------------------------------------------------------------------------------------
class GlycanParseError(ValueError):
    """Raised when a glycan string violates structural syntax and cannot be parsed."""

    pass


class GlycanParseWarning(UserWarning):
    """Raised when the parser employs heuristics or makes assumptions about an ambiguous structure."""

    pass


# Enums ----------------------------------------------------------------------------------------------------------------
class TokenType(Enum):
    BRANCH_START = auto()
    BRANCH_END = auto()
    LINKAGE = auto()
    SUGAR = auto()


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Token:
    type: TokenType
    value: bytes
    pos: int


@dataclass(slots=True, frozen=True)
class Substitution:
    modifier: int
    carbons: tuple[int, ...] = (0,)
    stoichiometry: float = 1.0


# Parsers --------------------------------------------------------------------------------------------------------------
class GlycanParser(ABC):
    _RE_LEXER = re.compile(rb"\]|\[|\([^)]+\)|[^\[\]()]+")

    def __init__(self) -> None:
        self.mono_map = dict(Sugars.name_to_index)
        for k, v in Sugars.alias_map.items():
            if v in self.mono_map:
                self.mono_map[k] = self.mono_map[v]

        self.mod_map = {
            b"Ac": self.mono_map[b"OAc"],
            b"OAc": self.mono_map[b"OAc"],
            b"Pyr": self.mono_map[b"Pyr"],
            b"P": self.mono_map[b"P"],
            b"PEtn": self.mono_map[b"PEtn"],
            b"OMe": self.mono_map[b"OMe"],
            b"Me": self.mono_map[b"OMe"],
            b"N": self.mono_map[b"N"],
            b"LAla": self.mono_map[b"LAla"],
            b"DAla": self.mono_map[b"DAla"],
            b"dAla": self.mono_map[b"DAla"],
            b"Ala": self.mono_map[b"Ala"],
            b"Fo": self.mono_map[b"Fo"],
            b"Lac": self.mono_map[b"Lac"],
            b"Gro": self.mono_map[b"Gro"],
            b"ol": self.mono_map[b"ol"],
        }

        mono_patterns = sorted(self.mono_map.keys(), key=len, reverse=True)
        mod_patterns = sorted(self.mod_map.keys(), key=len, reverse=True)

        self._RE_MOD = re.compile(rb"^(" + b"|".join(mod_patterns) + rb")")
        self._RE_MOD_SEARCH = re.compile(rb"(" + b"|".join(mod_patterns) + rb")")
        self._RE_MONO = re.compile(rb"^(" + b"|".join(mono_patterns) + rb")")

    def _lex(self, seq: bytes) -> Iterator[Token]:
        tokens = []
        for match in self._RE_LEXER.finditer(seq):
            val = match.group(0)
            pos = match.start()
            if val == b"]":
                tokens.append(Token(TokenType.BRANCH_START, val, pos))
            elif val == b"[":
                tokens.append(Token(TokenType.BRANCH_END, val, pos))
            elif val.startswith(b"("):
                tokens.append(Token(TokenType.LINKAGE, val[1:-1], pos))
            else:
                tokens.append(Token(TokenType.SUGAR, val.strip(), pos))
        yield from reversed(tokens)

    @abstractmethod
    def parse(self, sequence: bytes, strict: bool = False) -> Glycan:
        """Parse a canonical glycan sequence into a Glycan core model."""
        raise NotImplementedError
