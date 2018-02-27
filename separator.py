from skimage.measure import regionprops, label
from skimage.morphology import convex_hull_image
from skimage.segmentation import find_boundaries
import math
import numpy as np
import skimage
from itertools import combinations


def extendLineToMask(y1, x1, y2, x2, mask):
    if (y1 < 0) or (y2 < 0) or (x1 < 0) or (x2 < 0):  # input validation
        return 0, 0, 0, 0

    tc = (np.array([y1, x1]) + np.array([y2, x2])) / 2.0  # get center point

    # extend line to image bound
    if (x2 - x1 == 0):
        # is vertical line
        extendedLineY1 = 0
        extendedLineX1 = x1
        extendedLineY2 = mask.shape[0] - 1
        extendedLineX2 = x2
    else:
        # not vertical
        # calculate slope
        # use skimage line for better slope calculation
        rrP1, ccP1 = skimage.draw.line(int(y1), int(x1), int(y2), int(x2))
        m = ((ccP1 * rrP1).mean() - ccP1.mean() * rrP1.mean()) / ((ccP1 ** 2).mean() - (ccP1.mean()) ** 2)
        # calculate b
        b = y1 - (m * x1)

        extendedLineX1 = 0
        extendedLineY1 = m * extendedLineX1 + b  # y = mx+b
        # out of bound handling
        if (extendedLineY1 < 0) or (extendedLineY1 > mask.shape[0]):
            # get min/max Y
            extendedLineY1 = max(min(mask.shape[0] - 1, extendedLineY1), 0.0)
            # recalculate X
            extendedLineX1 = (extendedLineY1 - b) / m

        extendedLineX2 = mask.shape[1] - 1
        extendedLineY2 = m * extendedLineX2 + b
        # out of bound handling
        if (extendedLineY2 < 0) or (extendedLineY2 > mask.shape[0]):
            # get min/max Y
            extendedLineY2 = max(min(mask.shape[0] - 1, extendedLineY2), 0.0)
            # recalculate X
            extendedLineX2 = (extendedLineY2 - b) / m
            # check infinity
    if (math.isinf(extendedLineX1) or math.isinf(extendedLineX2) or math.isinf(extendedLineY1) or math.isinf(
            extendedLineY2)):
        return 0, 0, 0, 0

    # get extended line
    rrP1, ccP1 = skimage.draw.line(int(extendedLineY1), int(extendedLineX1), int(extendedLineY2), int(extendedLineX2))

    # get index of center point
    linecenterir = np.nonzero(rrP1 == int(tc[0]))[0]
    linecenteric = np.nonzero(ccP1 == int(tc[1]))[0]
    if (len(linecenteric) < len(linecenterir)):
        linecenteri = linecenteric
    else:
        linecenteri = linecenterir
    if (len(linecenteri)) > 0:
        # trim line to mask
        linecenterindex = linecenteri[0]
        try:  # there are some situations where issue occurs (line outside image) but I haven't had the time to add validation
            # get left and right part of the line (this is not actually direction of line but indexation of array)
            lineontheleft = mask[rrP1[0:linecenterindex], ccP1[0:linecenterindex]]
            lineontheright = mask[rrP1[linecenterindex:], ccP1[linecenterindex:]]
            # trim to mask
            lineonthelefttrimi = np.nonzero(lineontheleft == False)[0]
            lineontherighttrimi = np.nonzero(lineontheright == False)[0]
            # out of image bounds > better way will be to pad image with 1px False border, but this is faster
            if (len(lineonthelefttrimi) < 1) and (mask[rrP1[0], ccP1[0]] == True):
                lineonthelefttrimi = [-1]
            if (len(lineontherighttrimi) < 1) and (mask[rrP1[-1], ccP1[-1]] == True):
                lineontherighttrimi = [len(rrP1)]
            if (len(lineonthelefttrimi) > 0) and (len(lineontherighttrimi) > 0):
                lineonthelefttrim = lineonthelefttrimi[-1] + 1
                lineontherighttrim = linecenterindex + lineontherighttrimi[0]
                if (lineontherighttrim > lineonthelefttrim):
                    extendedLineY1 = rrP1[lineonthelefttrim:lineontherighttrim][0]
                    extendedLineX1 = ccP1[lineonthelefttrim:lineontherighttrim][0]
                    extendedLineY2 = rrP1[lineonthelefttrim:lineontherighttrim][-1]
                    extendedLineX2 = ccP1[lineonthelefttrim:lineontherighttrim][-1]
                else:
                    extendedLineY1 = 0
                    extendedLineX1 = 0
                    extendedLineY2 = 0
                    extendedLineX2 = 0
            else:
                # invalid line
                extendedLineY1 = extendedLineX1 = extendedLineY2 = extendedLineX2 = 0
        except:
            # invalid line
            extendedLineY1 = extendedLineX1 = extendedLineY2 = extendedLineX2 = 0
    else:
        # invalid line
        extendedLineY1 = extendedLineX1 = extendedLineY2 = extendedLineX2 = 0

    # return line coordinates
    return int(extendedLineY1), int(extendedLineX1), int(extendedLineY2), int(extendedLineX2)


# parallel line helper function
def parallelLine(px1, px2, offsetPixels, length=0.0):
    if (length == 0.0):
        length = math.sqrt((px1[1] - px2[1]) * (px1[1] - px2[1]) + (px1[0] - px2[0]) * (px1[0] - px2[0]))
    x1p = px1[1] + offsetPixels * (px2[0] - px1[0]) / length
    x2p = px2[1] + offsetPixels * (px2[0] - px1[0]) / length
    y1p = px1[0] + offsetPixels * (px1[1] - px2[1]) / length
    y2p = px2[0] + offsetPixels * (px1[1] - px2[1]) / length
    return [y1p, x1p], [y2p, x2p]


def splitValidation(px1, px2, img):
    # calculate line distance
    delta = (px1[1] - px2[1]) * (px1[1] - px2[1]) + (px1[0] - px2[0]) * (px1[0] - px2[0])
    if (delta > 0):
        L = math.sqrt(delta)
    else:
        L = 0
    # get top parallel line
    plpx1a, plpx2a = parallelLine(px1, px2, -4.0, length=L)
    # extend line to bound
    y1pea, x1pea, y2pea, x2pea = extendLineToMask(plpx1a[0], plpx1a[1], plpx2a[0], plpx2a[1], img)
    # get line length
    delta = (x1pea - x2pea) * (x1pea - x2pea) + (y1pea - y2pea) * (y1pea - y2pea)
    if (delta > 0):
        La = math.sqrt(delta)
    else:
        La = 0

    # get bottom parallel line
    plpx1b, plpx2b = parallelLine(px1, px2, 4.0, length=L)
    # extend line to bound
    y1peb, x1peb, y2peb, x2peb = extendLineToMask(plpx1b[0], plpx1b[1], plpx2b[0], plpx2b[1], img)
    # if top and bottom line are longer than split line
    delta = (x1peb - x2peb) * (x1peb - x2peb) + (y1peb - y2peb) * (y1peb - y2peb)
    if (delta > 0):
        Lb = math.sqrt(delta)
    else:
        Lb = 0

    plpx1a2, plpx2a2 = parallelLine(px1, px2, -2.0, length=L)
    # extend line to bound
    y1pea2, x1pea2, y2pea2, x2pea2 = extendLineToMask(plpx1a2[0], plpx1a2[1], plpx2a2[0], plpx2a2[1], img)
    # get line length
    delta = (x1pea2 - x2pea2) * (x1pea2 - x2pea2) + (y1pea2 - y2pea2) * (y1pea2 - y2pea2)
    if (delta > 0):
        La2 = math.sqrt(delta)
    else:
        La2 = 0

    # get bottom parallel line
    plpx1b2, plpx2b2 = parallelLine(px1, px2, 2.0, length=L)
    # extend line to bound
    y1peb2, x1peb2, y2peb2, x2peb2 = extendLineToMask(plpx1b2[0], plpx1b2[1], plpx2b2[0], plpx2b2[1], img)
    # if top and bottom line are longer than split line
    delta = (x1peb2 - x2peb2) * (x1peb2 - x2peb2) + (y1peb2 - y2peb2) * (y1peb - y2peb2)
    if (delta > 0):
        Lb2 = math.sqrt(delta)
    else:
        Lb2 = 0

    # if 5.0 line is bigger than split line and 2.0 line is bigger than 5.0 line
    # if -5.0 line is bigger than split line and -2.0 line is bigger than -5.0 line
    return (La >= L - 2) and (La >= La2) and (Lb >= L - 2) and (Lb >= Lb2)


def getCropMaskDimensions(img):
    cropmatrix = np.transpose(img.nonzero())
    cropcoordmin = np.amin(cropmatrix, axis=0)
    cropcoordmax = np.amax(cropmatrix, axis=0)
    cropy = cropcoordmin[0]
    cropx = cropcoordmin[1]
    cropy2 = cropcoordmax[0] + 1
    cropx2 = cropcoordmax[1] + 1
    return cropy, cropx, cropy2, cropx2


def separation(img):
    # make a copy
    inputimg_ = np.copy(img)

    # crop image to object bound > this helps speed up the processing
    cropy, cropx, cropy2, cropx2 = getCropMaskDimensions((inputimg_ == True))
    img_ = inputimg_[cropy:cropy2, cropx:cropx2]

    # define minimal convex area
    MIN_CONVEX_AREA = 5

    # calculate convex hull of object
    convexhull = convex_hull_image(img_)
    # invert convex hull
    convexhulldiff = img_ ^ convexhull
    # get boundaries > will help speed up processing
    boundaries = find_boundaries(convexhulldiff, connectivity=1, mode='inner')
    # split inset border objects
    label_img = label(boundaries, neighbors=8)
    if (label_img.max() > 1):
        # calculate region props
        regions = regionprops(label_img)
        # get candidates
        objectcandidates = []
        for props in regions:
            # limit to min convex area size
            if props.convex_area > MIN_CONVEX_AREA:
                y0, x0 = props.centroid  # this is not used
                orientation = props.orientation  # this is not used
                # convert image to points > will speed up processing
                objimg = label_img == props.label
                points_ = []
                for row_ in range(0, objimg.shape[0]):
                    for col_ in range(0, objimg.shape[1]):
                        if (objimg[row_][col_]):
                            points_.append([row_, col_])

                objectcandidates.append([orientation, points_])

        # get potential pairs
        for c in combinations(range(0, len(objectcandidates)), 2):
            # object A
            iobjA = c[0]
            # object B
            iobjB = c[1]
            # x,y border points for objects
            xy1 = np.array(objectcandidates[iobjA][1])
            xy2 = np.array(objectcandidates[iobjB][1])
            # numpy optimized way to find nearest points between two objects
            P = np.add.outer(np.sum(xy1 ** 2, axis=1), np.sum(xy2 ** 2, axis=1))
            N = np.dot(xy1, xy2.T)
            dists = np.sqrt(P - 2 * N)

            # find minimal distance pixel
            distsm_ = np.argmin(dists)
            # convert minimal pixel to x,y grid
            distsmp_ = divmod(distsm_, dists.shape[1])
            # convert to absolute position
            px1 = objectcandidates[iobjA][1][distsmp_[0]]
            px2 = objectcandidates[iobjB][1][distsmp_[1]]
            # separation line
            rr, cc = skimage.draw.line(px1[0], px1[1], px2[0], px2[1])
            line_ = img_[rr, cc]
            # check if line is valid = first check
            if (np.count_nonzero(line_[1:-1] == False) == 0):
                # check if split is valid = second check
                if (splitValidation(px1, px2, img_)):
                    img_[rr, cc] = 0
    # reconstruct image with cropped area
    inputimg_[cropy:cropy2, cropx:cropx2] = img_
    return inputimg_
