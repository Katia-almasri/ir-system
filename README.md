# ir-system
#for information retrieval using python<br>
steps for making IR system:<br>
1. text preprocssing <br>
2. making tfidf matrix<br>
3. process query (same as process the text)<br>
4. matchng & ranking (using cosing similarity)<br>
5. sort the documents from top 10 documents<br>
6. evaluation part: extract queries from the dataset<br>
7. extract the qrels from the datasets<br>
8. restructure the qrel <br>
9. for each query: get the retrieved documents and the relevant document for a specific query <br>
10. measure the (p@10, recall, MAP, MRR)<br>
11. add the extra feature (here query expansion)<br>
12. for each term in the query array: retrieve the similary terms and puth them in array<br>
13. rerun the system with the added feature and redo the steps(from 6 to 10)
and now you had an IR system (thats it!!)<br>
#steps for preprocessing the corpus:<br>
* remove white space <br>
* remove stopr words<br>
* stem_words <br>
* remove_lemma <br>
* remove_punctuation <br>
* text_lowercase <br>
* format_dates <br>
* process_shortcuts <br>




