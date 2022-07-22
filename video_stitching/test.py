import cv2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#cap_L = cv2.VideoCapture("../videos/vid_52/out_L.mp4")
cap_R = cv2.VideoCapture("../videos/vid_53/out_R.mp4")

#out_1 = cv2.VideoWriter("../videos/vid_52/out_L_new.mp4", fourcc, 60.0, (1920,  1080))
out_2 = cv2.VideoWriter("../videos/vid_53/out_R_new.mp4", fourcc, 60.0, (1920,  1080))
length = int(cap_R.get(cv2.CAP_PROP_FRAME_COUNT))
count = 0
#2635 2646
for i in range(1):
    count +=1
    #ret_L , frame_L = cap_L.read()
    ret_R , frame_R = cap_R.read()
    #print(count)
while(cap_R.isOpened() ):
    #ret_L , frame_L = cap_L.read()
    ret_R , frame_R = cap_R.read()
    if not (ret_R):
        break
    else:
        #cv2.imshow('L' , frame_L)
        cv2.imshow('R' , frame_R)
        
        #out_1.write(frame_L)
        out_2.write(frame_R)

        #cv2.waitKey(1)
        count += 1
        print(f"{round(count / length * 100 , 2)} %")
        #143 164
#out_1.release()
out_2.release()
#cap_L.release()
cap_R.release()
cv2.destroyAllWindows()