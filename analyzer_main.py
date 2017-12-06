from scripts.processing import Processor
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--train", type=str, default='files/mmhsct_dataset.csv', help="--train file_path")
parser.add_argument("--data", type=str, default='files/sr_all_comments_111.csv', help="--data file_path")
parser.add_argument("--output", type=str, default='files/system_a_output.csv', help="--output file_path")

args = parser.parse_args()
training_file = args.train
data_file = args.data
output_file = args.output

settings = {
    'training_file': training_file,
    'data_file': data_file,
    'max_reviews': None,  # Options: 0 to any integer | default: None (all)
    'output_file': output_file
}

if __name__ == "__main__":
    start_time = time.time()

    processor = Processor(settings=settings)
    processor.run()

    print("--- %s seconds ---" % (time.time() - start_time))