from datetime import datetime
from dateutil.relativedelta import relativedelta

from prefect import flow

import score


@flow
def income_prediction_backfill():
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()

    d = start_date

    while d < end_date:
        score.income_prediction(input_file, output_file, run_id)
        print(d)
        d += relativedelta(months=1)
        score.score(d)


if __name__ == '__main__':
    income_prediction_backfill()
