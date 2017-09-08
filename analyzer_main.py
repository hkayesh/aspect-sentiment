from processing import Processor


'''
#********** READ ME *********#
To run the program provide three file names and the number of maximum 
review comments to be processed. Files are 'training_file', 'data_file' and 'output_file'. 
Please save 'training_file' and 'data_file' in the same directory to this file. The system 
trains a classifier model using annotated data from 'training_file'. The model is then 
used to extract aspects from a specified number of review comments ('max_reviews') 
from 'data_file'. Finally, the result is saved as a CSV file in 'output_file'.
'''

settings = {
    'training_file': 'sr_dataset.csv',
    'data_file': 'sr_all_comments.csv',
    'max_reviews': 1000,  # Options: 0 to any integer | default: None (all)
    'output_file': 'srft.output.csv'
}

processor = Processor(settings=settings)
processor.run()
