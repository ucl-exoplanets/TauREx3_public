
def test_no_duplicates():
    from taurex._citation import taurex_citation
    from taurex.core import unique_citations_only

    entries = taurex_citation.citations()

    new_entries = [entries[0], entries[0], entries[0]]

    assert len(new_entries) == 3
    assert len(unique_citations_only(new_entries)) == 1
    assert unique_citations_only(new_entries)[0] == entries[0]




