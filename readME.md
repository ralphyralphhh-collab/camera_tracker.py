Camera tracker + identifier
 Detects objects from a webcam, identifies them (class name + confidence), and tracks each one with a persistent ID and a motion trail.  Install:     pip install ultralytics opencv-python  Run:     python camera_tracker.py              
   # default webcam (index 0)    
    python camera_tracker.py --source 1  
       # another camera     python camera_tracker.py --source video.mp4     python camera_tracker.py 
       classes person cat dog  
       Keys:  q = quit,  s = save snapshot

#1 first you need to download python 8.9 or higher, then put it in environmental variables

#2. next paste/clone the project on your IDE, make a foler and name the project "camera_tracker.py"

#3 then open terminal and type    "pip install ultralytics opencv-python"

#4 after its done downloading run the code using "python camera_tracker.py"