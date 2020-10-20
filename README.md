# Respiratory-Rate-Estimation
## Compiling and Running
1. Make sure to install all packages listed in requirements.txt. 
2. Using terminal, run "python RR_main.py -v (input video) -t (tracker)", you can omit the video argument if you intend to use another camera/webcam, the default tracker to track a regioin of interest (ROi) is the "boosting" tracker, however, you can choose from several available trackers to track the ROI.
3. Press "s" on the keyoard once a named window ("frame") appears on the screen.
4. Select a ROI by dragging the cursor diagonally from the top-left to the bottom-right to track and estimate respiratory rate.
5. Press spacebar.
6. Depending on the following paramaters (which can be modified), the time to display the first respiratory reading on the screen will vary.
  If you plan on estimating respiratory rate for a video, please make sure to tweak the following paramaters to ensure that you are getting an accurate estimation.
  a. Frames per second (F.P.S) of the video, which can be tweaked in line 108 of the "RR-main.py" script. In case you are using your camera to estimate pulse rate, please set F.P.S. according to how many frames the program is able to sample in a second as opposed to using the camera's frame sampling rate.
  b. Moving window: In the paper, we set window size in line 108 of the "RR-main.py" script as 30 seconds for an accurate estimation with the window stride being 1 second.
  c. Butter bandpass filter paramaters: Order, low-cut frequency and high-cut frequency (lines 119, 120 and 122) parameters. We find that the order parameter affects estimation accuracy and hence should be tuned according to the dataset.
