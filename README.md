matlab_code/singleSrcMixing.m - does mixing for single source 2 mic scenario;
		  
matlab_code/mixAllSingle.m - applies previous function to all source files in 'audio/source';  
    saves RIRs to the 'H' directory, and saves microphone signals to the 'audio/mics' directory;

stftAllSingle.py - finds stft for all mic signals

testAllSingleSrcTwoMic.py - tests all estimates against all truths

## Run order:
1. mixAllSingle.m
    
2. stftAllSingle.py
    window duration(in ms) can be specified in the last line:  
    
        singleSrcSTFT(m1_path, m2_path, WINDOW_DURATION)
    
3. testAllSingleSrcTwoMic.py


## test script
there is a test.sh which runs testAllSingleSrcTwoMic.py with different arguments  
following will do the job in bash:

    ./test.sh

result of one such run is given in txt_results/test_out.txt
