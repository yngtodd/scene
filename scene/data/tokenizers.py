from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


def spacy_splitter_factory():
    return SpacyWordSplitter(language='en_core_web_sm', pos_tags=False)


def spacy_word_tokenizer(x):
    return [w.text for w in spacy_splitter_factory().batch_split_words(x)]