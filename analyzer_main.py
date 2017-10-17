from processing import Processor
import time

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
    'training_file': 'mmh_dataset.csv',
    'data_file': '10k_data.csv',
    'max_reviews': 10,  # Options: 0 to any integer | default: None (all)
    'output_file': '10k_data.output.csv'
}


if __name__ == "__main__":
    start_time = time.time()

    processor = Processor(settings=settings)
    processor.run()

    print("--- %s seconds ---" % (time.time() - start_time))