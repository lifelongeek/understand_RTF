#OUT_FILE=txt_results/test_out.txt
for w in 20 50 100 200 400
do
    python testAllSingleSrcTwoMic.py --window_duration=$w

    for tp in 1 5 10 30 50 80
    do
	python testAllSingleSrcTwoMic.py --window_duration=$w --method="distribution_maxima"\
	       --top_percent=$tp
    done
done
