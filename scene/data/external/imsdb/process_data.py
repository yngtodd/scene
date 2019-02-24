import os
import collections
from typing import List

import numpy as np
import pandas as pd


def split_n_lines(script: str, num_chars: int=1000) -> List:
    return [script[i:i+num_chars] for i in range(0, len(script), num_chars)]


def read_script(script_path: str) -> str:
    """Read in IMSDB script as a string

    Parameters
    ----------
    script_path : str
        Path to the script 
    """
    with open(script_path, 'r') as f:
        file = f.read().splitlines()
    
    all_lines = []
    for line in file:
        line = " ".join(line.split())
        all_lines.append(line)
    
    all_lines = ' '.join(all_lines)
    return all_lines


def sample_from_script(script_path, num_lines, chars_per_line):
    """Sample num_lines from a script.
    
    Parameters
    ----------
    script_path : str
        Path to the script
    
    num_lines : int
        Number of lines to sample.
        
    chars_per_line : int
        Numer of consecutive characters considered a line.
        
    Returns
    -------
    lines : List
        All the sampled lines. 
    """
    script = read_script(script_path)
    script = split_n_lines(script, num_chars=chars_per_line)
    # sample with replacement since some scripts are sparse.
    lines = np.random.choice(script, num_lines, replace=True)
    return lines


def sample_from_genre(genre_path, num_lines, chars_per_line=1000):
    """Sample num_lines from a genre where each line contains chars_per_line.

    Parameters
    ----------
    genre_path : str
        Path to the script genres.
    
    num_lines : int
        Total number of lines to sample from genre.
    
    chars_per_line : int
        Number of characters that make up each line.
    """
    scripts = os.listdir(genre_path)
    # Randomly choose which scripts to pull from.
    script_idxs = np.random.randint(low=0, high=len(scripts)-1, size=num_lines)
    # Make sure we don't open the same script multiple times. 
    counter_scripts = collections.Counter(script_idxs)
    # idx: index indicating the script to open
    # n_lines: number of lines we should grab from the current script.
    all_lines = []
    for idx, n_lines in counter_scripts.items():
        script_path = os.path.join(genre_path, scripts[idx])
        lines = sample_from_script(script_path, n_lines, chars_per_line)
        all_lines.extend(lines)
    
    return all_lines


def create_genre_dataframe(genre, genre_path, num_lines, id_lead=9999):
    """Create a dataframe of randomly sampled script lines from a genre.

    Used to augment our dataset with IMSDB data.

    Parameters
    ----------
    genre : str
        Name of the genre. Must match `genre` field of original data
        Don't mispell things, *Todd*

    genre_path : str
        Path to all scripts of a given genre.

    num_lines : int
        Number of lines to sample from the genre.

    id_lead : int
        Identifier for IDs.
        Set to distinguish between augmented samples.

    Returns
    -------
    pandas dataframe : DataFrame['id', 'text', 'genre']
    """
    text = sample_from_genre(genre_path=genre_path, num_lines=num_lines)
    labels = [genre for x in range(len(text))]
    ids = [str(id_lead) + str(x) for x in range(len(text))]
    ids = [int(x) for x in ids]
    data = {'id': ids, 'text': text, 'genre': labels}
    return pd.DataFrame(data)