# aspect-sentiment
Instructions to run aspect-sentiment extraction:

Files required:

a. Training file: A CSV file having an annotated comment followed by corresponding
   aspect-sentiment pairs joined by a space in each line.
b. Data file: A CSV file having one comment per line


Note: All files should be in the project directory and all the commands should be
run from the project directory

Step 1: Run the stanford CoreNLP server using the following command

    $ python stanford_server.py

Step 2: Set data and training file name in settings variable of the 'analyzer_main.py' file.
    Change other settings if necessary.

Step 3: Run the following command start processing processing

    $ python analyzer_main.py


