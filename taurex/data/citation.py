from pybtex.database import Entry


def cleanup_string(string):
    return string.replace('{', '').replace('}', '').replace('\\', '')

def recurse_bibtex(obj, entries):
    for b in obj.__class__.__bases__:
        if issubclass(b, Citable):
            entries.extend(b.BIBTEX_ENTRIES)
            recurse_bibtex(b, entries)

def stringify_people(authors):

    return ', '.join([cleanup_string(str(p)) for p in authors])

def unique_citations_only(citations):

    current_citations = []
    for c in citations:
        if c not in current_citations:
            current_citations.append(c)
    return current_citations

def to_bibtex(citations):
    import uuid
    from pybtex.database import BibliographyData
    entries = {str(uuid.uuid4())[:8]: b for b in citations}
    bib_data = BibliographyData(entries=entries)

    return bib_data.to_string('bibtex')



def handle_publication(fields):
    journal = []
    if 'journal' in fields:
        journal.append(cleanup_string(fields['journal']))
    elif 'booktitle' in fields:
        journal.append(cleanup_string(fields['booktitle']))
    elif 'archivePrefix' in fields:
        journal.append(cleanup_string(fields['archivePrefix']))

    if 'volume' in fields:
        journal.append(cleanup_string(fields['volume']))
    elif 'eprint' in fields:
        journal.append(cleanup_string(fields['eprint']))
    if 'pages' in fields:
        journal.append(cleanup_string(fields['pages']))

    if 'month' in fields:
        journal.append(cleanup_string(fields['month']))

    if 'year' in fields:
        journal.append(cleanup_string(fields['year']))

    return ', '.join(journal)


def construct_nice_printable_string(entry, indent=0):

    mystring = ''
    indent = ''.join(['\t']*indent)
    form = f'{indent}%s\n'

    if 'title' in entry.fields:
        mystring += form % cleanup_string(entry.fields['title'])

    people = entry.persons
    if 'author' in people:
        mystring += form % stringify_people(people['author'])

    mystring += form % handle_publication(entry.fields)

    return mystring


class Citable:
    """
    Defines a class that contains citation
    information.
    """

    BIBTEX_ENTRIES = []
    """
    List of bibtext entries
    """
    def citations(self):
        entries = self.BIBTEX_ENTRIES[:]
        recurse_bibtex(self, entries)
        all_citations = [Entry.from_string(b, 'bibtex')
                         for b in entries]

        return unique_citations_only(all_citations)

    def nice_citation(self, prefix='', start_idx=0, indent=0):

        entries = self.citations()

        if len(entries) == 0:
            return ''

        return '\n'.join([construct_nice_printable_string(e)
                          for e in entries])
