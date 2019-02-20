import re


def clean_special_chars(x):
    """Remove special characters from text."""
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', x)
    return x


def clean_numbers(x):
    """Replace numbers with '#'.

    This is useful with using Word2Vec. 
    """
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x