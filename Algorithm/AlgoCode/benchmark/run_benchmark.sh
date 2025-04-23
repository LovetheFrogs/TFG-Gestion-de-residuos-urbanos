# Runs two benchmarks, one for 100 graphs between 40 and 120 nodes and another 
# one using TSPLib's graphs.

printf "Running TSPLib benchmark.\n"
python3 benchmark/benchmark.py -v 1 -m 1

printf "\nRunning randomized benchmark.\n"
python3 benchmark/benchmark.py -v 1 -m 0
