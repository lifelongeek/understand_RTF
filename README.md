singleSrcMixing.m - does mixing for single source 2 mic scenario;
		  
mixAllSingle.m - applies previous function to all source files in 'audio/source';
	       saves RIRs to the 'H' directory, and saves microphone signals
	       to the 'audio/mics' directory;

stftAllSingle.py - finds stft for all mic signals
testAllSingleSrcTwoMic.py - tests all estimates against all truths

Run order:
mixAllSingle.m
stftAllSingle.py
testAllSingleSrcTwoMic.py


# understand_RTF

1_audio_mixing.m

2_STFT.py

3_1_IPD_visualization.m

3_2_IMD_visualization.m
