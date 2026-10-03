"""Build a corpus: python3 -m corpora <target> [args]

    all          every lane, downloaded and generated - start here
    expand       turn generated .bin into the text files training reads
    fetch        any Hugging Face dataset, by name
    reasoning    OpenThoughts deliberation traces
    pg19         the PG19 test split, for a comparable bits-per-byte

    code         Python from the local filesystem   -> data_char
    arithmetic   synthesised, with scratchpads      -> data_math_char
    chat         primal chat and self-knowledge     -> data_chat_char
    chess        Lichess games in algebraic         -> data_chess_char

The four generators write train.bin, val.bin and meta.json into a data_*
directory: plain uint16 character streams, tied to no model, so a corpus
outlives any particular architecture. `expand` turns those into the text
files under data/ that `train.py read` actually opens - it skips .bin.

`all` runs the whole thing, in order, and is the only one most people need.
"""
import sys

BUILDERS = {"code": "corpora.code",
            "arithmetic": "corpora.arithmetic",
            "chat": "corpora.chat",
            "chess": "corpora.chess_games",
            "expand": "corpora.expand",
            "fetch": "corpora.fetch",
            "reasoning": "corpora.reasoning",
            "tool_use": "corpora.tool_use",
            "self_direction": "corpora.self_direction",
            "pg19": "corpora.pg19",
            "all": "corpora.build"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in BUILDERS:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    name = sys.argv.pop(1)
    import importlib
    return importlib.import_module(BUILDERS[name]).main()


if __name__ == "__main__":
    sys.exit(main())
