



from pybtex.database import Entry


def cleanup_string(string):
    return string.replace('{', '').replace('}', '').replace('\\','')


def stringify_people(authors):

    return ', '.join([cleanup_string(str(p)) for p in authors])

def handle_publication(fields):
    journal = []
    if 'journal' in fields:
        journal.append(cleanup_string(fields['journal']))
    elif 'booktitle' in fields:
        journal.append(cleanup_string(fields['booktitle']))

    if 'volume' in fields:
        journal.append(cleanup_string(fields['volume']))
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

    BIBTEX_ENTIRES=[]
    """
    List of bibtext entries
    """

    def citations(self, prefix='', start_idx=0):

        return [(f'{prefix}{self.__class__.__name__.lower()[:10]}-{idx+start_idx}', Entry.from_string(b, 'bibtex'))
                for idx, b in enumerate(self.BIBTEX_ENTIRES)]


