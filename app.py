from flask import Flask, redirect, url_for, render_template, request
import lemma_utils as lu
import corpus_repo

app = Flask(__name__)
cp = corpus_repo.CorpusRepo()
DEFAULT_SOURCE = 'fyj'

@app.route("/")
def home():
    return render_template("lemma_form.html", sentences='')


@app.route("/lemma", methods=['POST', 'GET'])
def lemma():
    if request.args is not None and len(request.args) > 0:
        output = get_sentences(request)
    else:
        output = ''
    return render_template("lemma_form.html", sentences=output)


def get_sentences(request):
    args = request.args
    if 'lemma' in args:
        lemma_searching_for = args['lemma']
    else:
        lemma_searching_for = None
    if 'max_examples' in args:
        max_examples = int(args['max_examples'])
    else:
        max_examples = lu.DEFAULT_MAX_EXAMPLES
    if 'source' in args:
        source = args['source']
    else:
        source = DEFAULT_SOURCE
    if source == 'md':
        sentences = lu.get_examples(cp.moby_sentences, cp.moby_lm, lemma_searching_for)
    elif source == 'fyj':
        sentences = lu.get_examples(cp.fyj_sentences, cp.fyj_lm, lemma_searching_for, max_examples)
    else:
        sentences = lu.get_examples(cp.fyj_sentences, cp.fyj_lm, lemma_searching_for, max_examples)
    output = ''
    for i in range(len(sentences)):
        output += str(i) + '. ' + sentences[i] + '\n\n'
    return output


if __name__ == "__main__":
    app.run()
