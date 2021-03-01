from taurex.core import Citable


class TauREXCitations(Citable):

    BIBTEX_ENTRIES = [
        """
        @misc{alrefaie2019taurex,
            title={TauREx III: A fast, dynamic and extendable framework for retrievals}, 
            author={Ahmed F. Al-Refaie and Quentin Changeat and Ingo P. Waldmann and Giovanna Tinetti},
            year={2019},
            eprint={1912.07759},
            archivePrefix={arXiv},
            primaryClass={astro-ph.IM}
        }
        """,

    ]


taurex_citation = TauREXCitations()
__citations__ = taurex_citation.nice_citation()