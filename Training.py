from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy


# new entity label
LABEL = 'New_tag'


TRAIN_DATA = [('How is Bitcoin doing today?', {'entities': [(15, 20, 'New_tag'), (21, 26, 'New_tag'), (7, 14, 'New_tag')]}), ('What People are talking about Bitcoin?', {'entities': [(5, 11, 'New_tag'), (16, 23, 'New_tag'), (30, 37, 'New_tag')]}), ('How Bitcoin will be doing tomorrow?', {'entities': [(4, 11, 'New_tag'), (26, 34, 'New_tag'), (20, 25, 'New_tag')]}), ('How Bitcoin will be doing one week?', {'entities': [(4, 11, 'New_tag'), (26, 34, 'New_tag'), (30, 34, 'New_tag')]}), ('How Bitcoin will be doing one month ahead?', {'entities': [(4, 11, 'New_tag'), (20, 25, 'New_tag'), (30, 35, 'New_tag'), (26, 35, 'New_tag'), (36, 41, 'New_tag')]}), ('What are the factors influencing Bitcoin prices?', {'entities': [(13, 20, 'New_tag'), (33, 40, 'New_tag'), (21, 32, 'New_tag')]})]



@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model='en', new_model_name='animal', output_dir='/Users/exepaul/', n_iter=20):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)

    nlp = spacy.load('en')


    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    ner.add_label(LABEL)   # add new entity label to entity recognizer

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update([text], [annotations], sgd=optimizer, drop=0.35,
                           losses=losses)
            print(losses)

    # test the trained model
    test_text = 'What People are talking about Bitcoin?'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == '__main__':
    plac.call(main)
