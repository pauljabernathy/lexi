

get_examples <- function(sentences, lemma_map, lemma, max_examples) {
  sentences[as.vector(lemma_map[lemma_map$word == lemma,4])[[1]][1:max_examples] + 1]
  # You have to add a 1 to the end because the indexing was done in Python, which indexes
  # arrays from 0, but R indexes them starting at 1.
}

format_examples_html <- function(examples) {
  result = "<ul>"
  for(i in 1:length(examples)) { 
    #  print(paste(paste("<li>", examples[i], sep=''), "</li>", sep='')) 
    #}
    result <- paste(result, examples[i], sep="<li>")
    result <- paste(result, "</li>", sep="")
  }
  result <- paste(result, "</ul>", sep="")
  result
}

something_else <- function() {
  result = "<ul>"
  result <- paste(result, "</ul>", sep="")
  result
}

library(reticulate)
print(getwd())
# need to be able to get a python version on the Shiny server!
#use_python("C:/Users/paulj_1e1uzlz/anaconda3/envs/tf/python.exe")
#use_python("C:/Users/paulj_1e1uzlz/anaconda3/python.exe")
fyj_lm <- data.frame(py_load_object("fyj_lemma_map.pkl"))
#fyj <- py_load_object("fyj.pkl")
sentences <- py_load_object("fyj_sentences.pkl")
# wc <- import("word_count")
utils <- import("lemma_utils")


examples <- get_examples(sentences, fyj_lm, "hablar", 20)
#print(examples)

html_examples <- format_examples_html(examples)
#print(html_examples)

s <- something_else()
print("s")


