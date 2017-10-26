from utilities import Utilities


class CombineSystems():
    def __init__(self):
        self.utilities = Utilities()

        self.storage_path = 'comment-level-datasets-2/'
        # self.storage_path = 'r-combine-outputs/'
        self.random_states = [111, 122, 133, 144, 155]
        self.categories = ['environment', 'waiting time', 'staff attitude professionalism', 'care quality', 'other']

    def combine_union(self, dataset_initials):
        for random_state in self.random_states:
            file_a = self.utilities.read_from_csv(self.storage_path + dataset_initials + '_output_' + str(random_state) + '.csv')
            # file_b = self.utilities.read_from_csv(self.storage_path + dataset_initials + '_output_' + str(random_state) + '.csv')
            file_b = self.utilities.read_from_csv('r-combine-outputs/' + dataset_initials + '_output_confidence_' + str(random_state) + '.csv')

            output = []
            for index, row_a in enumerate(file_a):
                row_b = file_b[index]

                comment = row_a[0]
                aspects = []

                # remove comment from the first column
                del row_a[0]
                del row_b[0]

                threshold = 0.8

                for a, b in zip(row_a, row_b):
                    if not a and not b and a in self.categories:
                        break

                    # union
                    # if a and a not in aspects:
                    #     aspects.append(a)
                    #
                    # if b and b not in aspects and b in self.categories:
                    #     aspects.append(b)

                    # union with threshold
                    if a and a not in aspects:
                        aspects.append(a)

                    if b and b.rsplit(' ', 1)[0] in self.categories and b.rsplit(' ', 1)[0] not in aspects and float(b.rsplit(' ', 1)[1]) >= threshold:
                        print comment
                        print b.rsplit(' ', 1)[0]

                        aspects.append(b.rsplit(' ', 1)[0])



                        # intersection
                    # if a and b and a == b and a not in aspects:
                    #     aspects.append(a)

                if len(aspects) < 1:
                    aspects = ['other']

                output.append([comment] + aspects)

            self.utilities.save_list_as_csv(output, 'r-combine-outputs/' + dataset_initials + '_combined_confidence_' + str(random_state) + '.csv')


combine = CombineSystems()
combine.combine_union('mmhsct')