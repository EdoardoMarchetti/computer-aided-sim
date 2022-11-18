import pandas as pd
from datetime import date


DAYS_IN_MONTHS = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31
}


def date_to_int(date: date) -> int:
    day_int = date.day
    for i in range(1, date.month):
        day_int += DAYS_IN_MONTHS[i]
    return day_int


def main() -> None:
    df = pd.read_csv('births_data.csv')
    # remove leap years
    df = df[(df['month']!=2) | (df['date_of_month']!=29)]
    df['full_date'] = df.apply(
        lambda row: date(
            year=row['year'],
            month=row['month'],
            day=row['date_of_month']
            ),
            axis=1
            )
    df['day_number'] = df.apply(
        lambda row: date_to_int(row['full_date']),
        axis=1
        )
    avg_by_day = df[['births', 'day_number']]\
        .groupby('day_number').mean()
    avg_births_per_year = df[['year', 'births']].groupby('year').sum().mean()
    print(avg_births_per_year)
    avg_by_day['probability'] = avg_by_day.apply(
        lambda row: row['births'] / avg_births_per_year,
        axis=1
        )
    print(avg_by_day.head())
    print(avg_by_day['probability'].sum())
    avg_by_day['probability'].to_csv('estimate_realistic_distr.csv')
    

if __name__ == '__main__':
    main()
