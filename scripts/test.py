import cv2


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.putText(frame, f"This is frame No. 1", (30, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255), 3)
        cv2.imshow("Testing Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
main()