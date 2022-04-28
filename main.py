from math import cos, sin, sqrt
from typing import List, Optional, Tuple
from cv2 import VideoCapture, cvtColor, COLOR_BGR2HSV, split, adaptiveThreshold, ADAPTIVE_THRESH_GAUSSIAN_C, \
    THRESH_BINARY, imshow, Canny, HoughLines, line, LINE_AA, waitKey, \
    drawContours, findContours, RETR_TREE, CHAIN_APPROX_SIMPLE, approxPolyDP, contourArea, arcLength, \
    FONT_HERSHEY_COMPLEX, putText, medianBlur, COLOR_BGR2GRAY, GaussianBlur, ADAPTIVE_THRESH_MEAN_C, \
    CHAIN_APPROX_NONE, moments, HoughCircles, HOUGH_GRADIENT, circle, COLOR_GRAY2BGR
from numpy import pi, ndarray, around, concatenate, uint16

TRIANGLE = 3
QUADRANGLE = 4
COLOR_RED = (0, 0, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 0, 0)
minContourArea = 1500  # najmensia plocha kontury
maxCentersDistance = 50  # maximalna vzdialenost stredov
lowThresh = 170
lowThresh2 = 150
highThresh = 255
houghParam1 = 80
houghParam2 = 32

video = VideoCapture(0)

if not video.isOpened():
    raise IOError("Cannot open webcam")


def is_triangle(cor) -> bool:
    return len(cor) == 3


def is_quadrangle(cor) -> bool:
    return len(cor) == 4


def check_true_coordinates(contour_coordinates, hough_coordinates, delta=20) -> List:
    coordinates = []
    for h_cor in hough_coordinates:
        for con in contour_coordinates:
            dx = abs(con[0] - round(h_cor[0]))
            dy = abs(con[1] - round(h_cor[1]))
            if delta > dx and delta > dy:
                coordinates.append((round(h_cor[0]), round(h_cor[1])))

    return list(set(coordinates))


def unification(ok_coordinates, not_ok_coordinates, delta=20) -> Optional[Tuple]:
    save_p = []
    for not_ok_coordinate in not_ok_coordinates:
        dx = abs(ok_coordinates[0] - not_ok_coordinate[0])
        dy = abs(ok_coordinates[1] - not_ok_coordinate[1])
        if delta > dx and delta > dy:
            save_p.append(not_ok_coordinate)

    x = sum(i[0] for i in save_p)
    y = sum(i[1] for i in save_p)

    if save_p:
        x = x / len(save_p)
        y = y / len(save_p)
        return x, y
    else:
        return None


def coordinates_from_approx(approx_for_save_coordinates) -> List[List]:
    coordinates = []
    for par in approx_for_save_coordinates:
        coordinates.append([par[0][0], par[0][1]])
    return coordinates


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def change_threshold_for_hough(s_lines, threshold):
    if len(s_lines) > 12 and threshold < 200:
        threshold = threshold + 10
    elif len(s_lines) < 8 and threshold > 50:
        threshold = threshold - 10
    return threshold


def find_quadrangle(approx, frame, coordinates):
    save_coordinates_for_square = coordinates_from_approx(approx)
    cor = check_true_coordinates(save_coordinates_for_square, coordinates)
    final_cords = []
    for cord in save_coordinates_for_square:
        rounded_cords = unification(cord, cor)
        if rounded_cords:
            final_cords.append(rounded_cords)

    sq = list(set(final_cords))
    if len(sq) == QUADRANGLE:
        sort_x = sorted(sq, key=lambda x: x[0], reverse=True)
        line(frame, (round(sort_x[0][0]), round(sort_x[0][1])), (round(sort_x[1][0]), round(sort_x[1][1])), COLOR_BLACK,
             5, LINE_AA)
        line(frame, (round(sort_x[2][0]), round(sort_x[2][1])), (round(sort_x[3][0]), round(sort_x[3][1])), COLOR_BLACK,
             5, LINE_AA)
        width = sort_x[0][0] - sort_x[3][0]
        sort_y = (sorted(sq, key=lambda x: x[1], reverse=True))
        line(frame, (round(sort_y[0][0]), round(sort_y[0][1])), (round(sort_y[1][0]), round(sort_y[1][1])), COLOR_BLACK,
             5, LINE_AA)
        line(frame, (round(sort_y[2][0]), round(sort_y[2][1])), (round(sort_y[3][0]), round(sort_y[3][1])), COLOR_BLACK,
             5, LINE_AA)
        height = sort_y[0][1] - sort_y[3][1]
        aspectRatio = width / height
        if 0.9 <= aspectRatio <= 1.1:
            putText(frame, "Square", (round(sort_x[0][0]), round(sort_x[0][1])), FONT_HERSHEY_COMPLEX, 1.5, COLOR_BLACK)
        else:
            putText(frame, "Rectangle", (round(sort_x[0][0]), round(sort_x[0][1])), FONT_HERSHEY_COMPLEX, 1.5,
                    COLOR_BLUE)


def find_triangles(approx, frame, intersection):
    save_coordinates_for_triangle = coordinates_from_approx(approx)
    cor = check_true_coordinates(save_coordinates_for_triangle, intersection)
    final_cords = []
    for cord in save_coordinates_for_triangle:
        rounded_cords = unification(cord, cor)
        if rounded_cords:
            final_cords.append(rounded_cords)

    tr = list(set(final_cords))
    if len(tr) == TRIANGLE:
        line(frame, (round(tr[0][0]), round(tr[0][1])), (round(tr[1][0]), round(tr[1][1])), COLOR_BLACK, 5, LINE_AA)
        line(frame, (round(tr[1][0]), round(tr[1][1])), (round(tr[2][0]), round(tr[2][1])), COLOR_BLACK, 5, LINE_AA)
        line(frame, (round(tr[2][0]), round(tr[2][1])), (round(tr[0][0]), round(tr[0][1])), COLOR_BLACK, 5, LINE_AA)
        sort_y = (sorted(tr, key=lambda x: x[1], reverse=True))
        putText(frame, "Triangle", (round(sort_y[2][0]), round(sort_y[2][1])), FONT_HERSHEY_COMPLEX, 1.5, COLOR_BLACK)


def collect_intersections(lines) -> List:
    inters = []
    for first_line in lines:
        for second_line in lines:
            inter = line_intersection(first_line, second_line)
            if inter:
                inters.append(inter)
    return inters


def collect_lines(h_lines, hough_copy) -> List:
    save_lines = []
    for h_line in h_lines:
        rho = h_line[0][0]
        theta = h_line[0][1]
        a = cos(theta)
        b = sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        line(hough_copy, pt1, pt2, COLOR_RED, 3, LINE_AA)
        save_lines.append([pt1, pt2])
    return save_lines


def main():
    threshold = 100

    while True:
        #########################################################################################
        #                                   # SETUP #
        #########################################################################################
        ret, frame = video.read()
        hough_copy = frame.copy()
        copy_contour = frame.copy()
        img = frame.copy()

        #########################################################################################
        #                          # setup for detected circle #
        #########################################################################################
        imgGray = cvtColor(img, COLOR_BGR2GRAY)
        imgBlur = GaussianBlur(imgGray, (7, 7), 0)  # 15
        # HSV model
        imgHSV = GaussianBlur(cvtColor(img, COLOR_BGR2HSV), (7, 7), 0)
        imgS = imgHSV[:, :, 1].copy()
        # Canny filter
        threshold1 = 50
        threshold2 = 150
        imgCanny = Canny(imgBlur, threshold1, threshold2)
        # detekcia pomocou kontur
        thresh = adaptiveThreshold(imgS, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 1023, -115)  # adaptivny threshold
        #########################################################################################

        #########################################################################################
        #                      # setup for detected triangle and quadrangle #
        #########################################################################################
        img = cvtColor(hough_copy, COLOR_BGR2HSV)
        h, s, v = split(img)
        th2 = adaptiveThreshold(s, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 1023, -105)
        thresh_for_line = medianBlur(th2, 9)
        canny2 = Canny(thresh_for_line, 220, 250)
        #########################################################################################

        #########################################################################################
        #                                 # DATA COLLECT #
        #########################################################################################
        h_lines: ndarray = HoughLines(canny2, 1, pi / 180, threshold, 0, 0)
        s_lines = []

        if h_lines is not None:
            s_lines = collect_lines(h_lines, hough_copy)
            threshold = change_threshold_for_hough(s_lines, threshold)
        inters = collect_intersections(s_lines)

        contours_for_circle, _ = findContours(thresh, RETR_TREE, CHAIN_APPROX_NONE)
        contours_for_line, _ = findContours(thresh_for_line, RETR_TREE, CHAIN_APPROX_SIMPLE)

        #########################################################################################

        #########################################################################################
        #                                   # DATA PROCESSING #
        #########################################################################################

        #########################################################################################
        #                         # data processing for triangle and quadrangle #
        #########################################################################################
        for contour in contours_for_line:
            approx = approxPolyDP(contour, 0.02 * arcLength(contour, True), True)
            area = contourArea(contour)
            if area < minContourArea:
                continue
            drawContours(image=copy_contour, contours=contours_for_line, contourIdx=-1, color=(0, 255, 0), thickness=2,
                         lineType=LINE_AA)
            if is_triangle(approx):
                find_triangles(approx, frame, inters)

            if is_quadrangle(approx):
                find_quadrangle(approx, frame, inters)
        ########################################################################################

        ########################################################################################
        #                            # data processing for circle #
        ########################################################################################
        foundContours = []
        for contour in contours_for_circle:

            shape = approxPolyDP(contour, 0.01 * arcLength(contour, True), True)
            x_cor = shape.ravel()[0]
            y_cor = shape.ravel()[1]

            if len(shape) > 10:  # detekcia kruhu
                area = contourArea(contour)
                if area < minContourArea:  # kontura je prilis mala
                    continue

                drawContours(img, [shape], 0, (255, 0, 0), 4)
                putText(img, "Kruh", (x_cor, y_cor), FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                sumX = 0
                sumY = 0

                M = moments(contour)  # najdenie stredu
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                circle(img, (x, y), 2, (255, 255, 0), 5)  # stredy najdenych objektov
                foundContours.append([x, y, area])

        detected = False
        if foundContours is not None:  # ak nie su najdene kontury, tak je toto zbytocne
            # Houghova transformacia
            circles = HoughCircles(image=imgCanny, method=HOUGH_GRADIENT, dp=1, minDist=150, param1=houghParam1,
                                   param2=houghParam2, minRadius=15, maxRadius=200)

            if circles is not None:  # su detekovane nejake kruhy
                circles = uint16(around(circles))
                for i in circles[0, :]:
                    # vykreslenie kruhu
                    circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # vykreslenie stredu
                    circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
                    for x, y, area in foundContours:
                        distance = sqrt((i[0] - x) ** 2 + (i[1] - y) ** 2)
                        if distance < maxCentersDistance:
                            circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                            circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
                            detected = True
        ########################################################################################

        ########################################################################################
        #                             # DATA VISUALISATION #
        ########################################################################################
        imgCannyRGB = cvtColor(imgCanny, COLOR_GRAY2BGR)
        threshRGB = cvtColor(thresh, COLOR_GRAY2BGR)
        imgSRGB = cvtColor(imgS, COLOR_GRAY2BGR)
        imgCanny2RGB = cvtColor(canny2, COLOR_GRAY2BGR)
        numpy_horizontal_concat = concatenate((imgCannyRGB, threshRGB, img, imgSRGB), axis=1)
        numpy_horizontal_concat_2 = concatenate((imgCanny2RGB, copy_contour, hough_copy, frame), axis=1)
        imshow("image", numpy_horizontal_concat)
        imshow("image2", numpy_horizontal_concat_2)
        #########################################################################################
        k = waitKey(30) & 0xff
        if k == 27:
            break


if __name__ == '__main__':
    main()
