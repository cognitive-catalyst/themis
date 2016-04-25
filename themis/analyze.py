from themis import CsvFileType, QUESTION, ANSWER, CONFIDENCE, IN_PURVIEW, CORRECT, FREQUENCY

SYSTEM = "System"


class CollatedFileType(CsvFileType):
    columns = [QUESTION, SYSTEM, ANSWER, CONFIDENCE, IN_PURVIEW, CORRECT, FREQUENCY]

    def __init__(self):
        super(self.__class__, self).__init__(CollatedFileType.columns)

    @staticmethod
    def output_format(collated):
        collated = collated[CollatedFileType.columns]
        collated = collated.sort_values([QUESTION, SYSTEM])
        return collated.set_index([QUESTION, SYSTEM, ANSWER])
