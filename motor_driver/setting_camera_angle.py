import cv2
import time


def setting_camera_angle(arduino_control, source=0, flip=True, left_angle=-45, right_angle=45):

    def show_info(word, angle):
        print("Press 'q' to quit, 'Enter' to continue")
        print("       w(+5)")
        print("a(-1)  s(+5)  d(+1)")
        print("Setting {} angle, current is {:-4d}".format(word, angle))

    def setting_angle(arduino_control, k, step, print_info, left_angle, right_angle):
        if step == 0:
            angle = left_angle
            word = "left"
            if print_info:
                for i in range(0, angle - 1, -3):
                    print(i)
                    arduino_control.write(i)
                    time.sleep(0.1)
                arduino_control.write(angle)
                show_info("left", angle)
                print_info = False
        else:
            angle = right_angle
            word = "right"
            if print_info:
                for i in range(left_angle, angle + 1, 3):
                    arduino_control.write(i)
                    time.sleep(0.1)
                arduino_control.write(angle)
                show_info("left", angle)
                print_info = False

        old_angle = angle

        if k == ord('d'):
            angle += 1
        elif k == ord('a'):
            angle -= 1
        elif k == ord('w'):
            angle += 5
        elif k == ord('s'):
            angle -= 5
        elif k == ord('\r'):
            print_info = True
            step += 1

        if old_angle != angle:
            arduino_control.write(angle)
            if step == 0:
                print("seting 'left' angle to {}".format(angle))
                left_angle = angle
            else:
                print("seting 'right' angle to {}".format(angle))
                right_angle = angle
        return step, print_info, left_angle, right_angle

    # define a video capture object
    vid = cv2.VideoCapture(source)

    # 0 -> setting left angle
    # 1 -> setting right angle
    step = 0
    # print information
    print_info = True

    while(True):
        ret, frame = vid.read()
        if flip:
            frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

        if step < 2:
            step, print_info, left_angle, right_angle = setting_angle(
                arduino_control, k, step, print_info, left_angle, right_angle)
        else:
            break

    vid.release()
    cv2.destroyAllWindows()

    print("\nSetting successfully!")
    print("Setting left  angle to {:-4d}".format(left_angle))
    print("Setting right angle to {:-4d}".format(right_angle))

    return left_angle, right_angle
