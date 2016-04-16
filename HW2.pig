clear;
raw_data = LOAD '$input'  USING PigStorage(',') AS (
    Year: int,
    Month: int,
    DayofMonth: int,
    DayOfWeek: int,
    DepTime: int,
    CRSDepTime: int,
    ArrTime: int,
    CRSArrTime: int,
    UniqueCarrier: chararray,
    FlightNum: int,
    TailNum: chararray,    
    ActualElapsedTime: int,
    CRSElapsedTime: int,
    AirTime: int,
    ArrDelay: int,
    DepDelay: int,
    Origin: chararray,
    Dest: chararray,
    Distance: int,
    TaxiIn: int,
    TaxiOut: int,
    Cancelled: int,
    CancellationCode: chararray,
    Diverted: int,
    CarrierDelay: int,
    WeatherDelay: int,
    NASDelay: int,
    SecurityDelay: int,
    LateAircraftDelay: int 
);
-- Eliminate csv header
data = FILTER raw_data BY UniqueCarrier != 'UniqueCarrier';
-- DUMP data;

/* 
    Compute avg delay for each year 
    "SELECT AVG(DepDelay) FROM `data` GROUP BY `Year`
*/
group_year = GROUP data BY Year;
-- DUMP group_year;
avg_delay = FOREACH group_year GENERATE group, AVG(data.DepDelay);
DUMP avg_delay;
STORE avg_delay INTO 'output/1avg_delay' USING PigStorage(',');

/*
    Find max delay each yaer
*/
max_delay = FOREACH group_year GENERATE group, MAX(data.DepDelay);
DUMP max_delay;
STORE max_delay INTO 'output/2max_delay' USING PigStorage(',');

/*
    count & average of delay caused by weather factors
*/
delay_by_weather = FILTER data BY WeatherDelay > 0;
delay_by_weather_group = GROUP delay_by_weather ALL;
--delay_by_weather = ORDER delay_by_weather BY Year;
delay_by_weather_count = FOREACH delay_by_weather_group GENERATE COUNT(delay_by_weather);
delay_by_weather_avg = FOREACH delay_by_weather_group GENERATE AVG(delay_by_weather.WeatherDelay);
DUMP delay_by_weather;
DUMP delay_by_weather_count;
DUMP delay_by_weather_avg;
--STORE delay_by_weather INTO 'output/3delay_by_weather' USING PigStorage(',');
STORE delay_by_weather_count INTO 'output/3delay_by_weather_count' USING PigStorage(',');
STORE delay_by_weather_avg INTO 'output/4delay_by_weather_avg' USING PigStorage(',');


/*
    Get least delay time for each year
*/
schedule_dep = GROUP data BY (Year, CRSDepTime);
year_delay = FOREACH schedule_dep GENERATE group.$0, group.$1, AVG(data.ArrDelay);
year_delay_group = GROUP year_delay BY $0;
min_delay = FOREACH year_delay_group GENERATE group, MIN($1.$2);
d0 = JOIN year_delay BY $2, min_delay BY $1;
d1 = JOIN year_delay BY $2, min_delay BY $1;
result = JOIN d0 BY $0, d1 BY $3;
least_delay_by_year = FOREACH result GENERATE $0, $1, $2;
least_delay_by_year = DISTINCT least_delay_by_year;
DUMP least_delay_by_year;
STORE least_delay_by_year INTO 'output/5least_delay_by_year' USING PigStorage(',');


