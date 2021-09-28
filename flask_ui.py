from flask import Flask, redirect, url_for, render_template, request
import lemma_utils as lu
import corpus_repo

app = Flask(__name__)
cp = corpus_repo.CorpusRepo()


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
    filters = {}
    if 'lemma' in args:
        lemma_searching_for = args['lemma']
    else:
        lemma_searching_for = None
    for k in args.keys():
        if 'name' in k and k.replace('name', 'value') in args.keys() and args[k] is not None and args[k] != '':
            filters[args[k]] = args[k.replace('name', 'value')]
    # sentences = lu.search_sentences_for_examples(cp.fyj_sentences, cp.fyj_lm, lemma_searching_for, filters)
    # using lu.get_examples for the time being to avoid using the huge spacy files; functionality will be reduced
    sentences = lu.get_examples(cp.fyj_sentences, cp.fyj_lm, lemma_searching_for)
    #sentences = lu.get_examples(cp.moby_sentences, cp.moby_lm, lemma_searching_for)
    output = ''
    for i in range(len(sentences)):
        output += str(i) + '. ' + sentences[i] + '\n\n'
    return output

if __name__ == "__main__":
    app.run()
