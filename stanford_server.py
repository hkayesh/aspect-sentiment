import os

print "Starting Stanford CoreNLP server...."
os.system("cd resources/stanford-corenlp-full-2016-10-31;java -mx5g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 30000")
