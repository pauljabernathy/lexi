from flask import Flask, redirect, url_for, render_template, request
import lemma_utils as lu
import corpus_repo

app = Flask(__name__)

cp = corpus_repo.CorpusRepo()

@app.route("/")
def home():
    #return '¡Hola mundo!'
    return render_template("index.html", content=["one", "two", "three"])

@app.route("/<name>")
def user(name):
    #return f"Hello <i>{name}</i>"
    return render_template("index.html", content=name)

@app.route("/admin")
def admin():
    return redirect(url_for("user", name="Admin"))

@app.route("/lemma", methods=['POST', 'GET'])
def lemma():
    bp = 'bp'
    #if request.method == 'POST':
    #    pass
    #else:
    args = request.args
    filters = {}
    if 'lemma' in args:
        lemma = args['lemma']
    else:
        lemma = None
    for k in args.keys():
        #filters[k] = args[k]
        if 'name' in k and k.replace('name', 'value') in args.keys() and args[k] is not None and args[k] != '':
            filters[args[k]] = args[k.replace('name', 'value')]
    #lu.search_doc_for_examples(cp.fyj, lemma, filters)
    sentences = lu.search_sentences_for_examples(cp.fyj_sentences, cp.fyj_lm, lemma, filters)
    output = ''
    for i in range(len(sentences)):
        #output += s + '\n'
        output += str(i) + '. '  + sentences[i] + '\n\n'
    sentences.__format__()
    len(sentences)
    return render_template("lemma_form.html", sentences=output)

if __name__ == "__main__":
    app.run()