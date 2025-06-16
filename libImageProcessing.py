import numpy
from PIL import Image

# convertToBlackAndWhite(imgIN) ----- ----- ----- ----- ----- 
# OBJECTIVE:    Convert image to black and white;
# INPUT(S):     imgIN: target image; 
# OUTPUT(S):    imgBwOUT: converted image;

def convertToBlackAndWhite(imgIN):

    # ASDc: Convert image to 8-bit grayscale via PIL convert() method.
    # NOTE: That "L" defines output as 1-bit/pixed grayscale image.
    # NOTE: https://www.codecademy.com/resources/docs/pillow/image/convert
    # NOTE: That "dither" attribute determined dithering employed.
    imgGrayScale = imgIN.convert("L")
    imgGrayScale.show()
    
    # ASDc: Convert image to rank-2 matrix, containing integer values 0 (black) to 255 (white).
    matImgGrayScale = numpy.array(imgGrayScale)
    
    # ASDc: Define threshold as divider between black and white pixels.
    thresholdBW = 100
    
    # ASDc: Convert grayscale image-matrix to boolean type, where true=white and false=black.
    matImgBoolBW = matImgGrayScale < thresholdBW
    
    # ASDc: Convert boolean matrix to one containing values between false=0 and true=255 (aka. grayscale).
    matImgBW = matImgBoolBW.astype(numpy.uint8) * 255 # NOTE: That .astype() casts booleans to integers.

    # ASDc: Flip all values such that 0->255 and 255->0.
    matImgBW = 255 - matImgBW

    # ASDc: Convert matrix-image to image.
    imgBwOUT = Image.fromarray(matImgBW)

    # ASDc: Return imgEdgeBW.
    return imgBwOUT

# FUNCTION appleEdgeDetection(imgIN) ----- ----- ----- ----- -----
# OBJECTIVE:    Return image with edges detected and fill removed;
# INPUT(S):     imgIN: image to which edge-detection is applied;
# OUTPUT(S):    imgEdgeBW: image with edges detected and fill removed;

def applyEdgeDetection(imgIN):

    # ASDc: Convert image to 8-bit grayscale via PIL convert() method.
    # NOTE: That "L" defines output as 1-bit/pixed grayscale image.
    # NOTE: https://www.codecademy.com/resources/docs/pillow/image/convert
    # NOTE: That "dither" attribute determined dithering employed.
    imgGrayScale = imgIN.convert("L")
    
    # ASDc: Convert image to rank-2 matrix, containing integer values 0 (black) to 255 (white).
    matImgGrayScale = numpy.array(imgGrayScale)
    
    # ASDc: Define threshold as divider between black and white pixels.
    thresholdBW = 100
    
    # ASDc: Convert grayscale image-matrix to boolean type, where true=white and false=black.
    matImgBoolBW = matImgGrayScale < thresholdBW
    
    # ASDc: Convert boolean matrix to one containing values between false=0 and true=255 (aka. grayscale).
    matImgBW = matImgBoolBW.astype(numpy.uint8) * 255 # NOTE: That .astype() casts booleans to integers.

    # ASDc: Create zero-matrix with same shape as matImgBW.
    matEdgeBW = numpy.zeros_like(matImgBW)

    # ASDc: Calculate three matrices of same size, as follows:
    # 1. matNoFirstkRowOrFirstCol: matEdgeBW without first row and first column;
    # 2. matNoLastkRowOrFirstCol: matEdgeBW without last row and first column;
    # 3. matNoFirstkRowOrLastCol: matEdgeBW without first row and last column;
    # NOTE: That one may compare these three matrices to find edges within the original image.
    matNoFirstkRowOrFirstCol = matImgBW[1:, 1:]
    matNoLastkRowOrFirstCol = matImgBW[:-1, 1:]
    matNoFirstkRowOrLastCol = matImgBW[1:, :-1]

    # ASDc: Compare second and third matrices (as defined above) to the first, with edges being identified by mismatches between them.
    # NOTE: This works because removing the first row/column essentially shifts all elements up/left by one, and removing the last row/column eliminates the leftover/additional row/column.
    # NOTE: That & serves as a bit-wide AND operator, with | representing the OR operator.
    matEdgeBW[1:, 1:] = (
        ((matNoFirstkRowOrFirstCol == matNoFirstkRowOrLastCol) & (matNoFirstkRowOrFirstCol == matNoLastkRowOrFirstCol))
        .astype(numpy.uint8) * 255
    )

    # ASDc: Delete first and last rows/columns from matEdgeBW to remove artifacts from edge detection.
    matEdgeBW = matEdgeBW[1:-1, 1:-1]

    # ASDc: Convert matrix-image to image.
    imgEdgeBW = Image.fromarray(matEdgeBW)

    # ASDc: Return imgEdgeBW.
    return imgEdgeBW

# FUNCTION getNonWhiteBounds(imgIN) ----- ----- ----- ----- -----
# OBJECTIVE:    Accept image (most likely to which edge-detection has been applied) and return boundaries of any non-white pixels;
# INPUT(S):     imgIN: image to be analyzed;
# OUTPUT(S):    tupleImageBounds: tuple containing boundaries of non-white pixels in the form {left, top, right, bottom};

def getNonWhiteBounds(imgIN):

    # ASDc: Convert image to matrix representation.
    matImgIN = numpy.array(imgIN)
    # NOTE: That an alternatve method is shown below.
    # seqImgIN = imgIN.getdata()
    # matImgIN = list(seqImgIN) 

    # ASDc: Get image size in terms of pixels.
    [ imgWidth, imgHeight ] = imgIN.size

    # ASDc: Create empty lists to store indices of rows and columns that contain at least one non-white pixel.
    listNonWhitekRows = []
    listNonWhiteCols = []

    # ASDc: Cycle through all rows within image.
    for kRowK in range(imgHeight):

        # ASDc: Cycle through all columns within image.
        for colK in range(imgWidth):

            # ASDc: Get pixel value at (kRowK, colK).
            pixK = imgIN.getpixel((colK, kRowK))

            # ASDc: Calculate minimum pixel value contained within pixK, indicating its maximum darkness.
            # NOTE: That if-statement allows algorithm to handle images with any number of channels, from 1 (grayscale) to 3+ (RGB).
            if isinstance(pixK, (tuple, list)): # NOTE: pixK is tuple containing multiple values.
                minPixK = min(pixK)
            else:
                minPixK = pixK # NOTE: pixK is integer.
            
            # ASDc: Check if pixel is non-white.
            # NOTE: That all tuple values must be 255 for pixel to be considered 100% white.
            if ( minPixK != 255 ):
                
                # ASDc: Append current row and column indices to respective lists, indicating those with at least one non-white pixel.
                listNonWhiteCols.append(colK)
                listNonWhitekRows.append(kRowK)
                
    # ASDc: Extract boundaries of non-white image by taking maximum and minimum values from listNonWhitekRows and listNonWhiteCols.
    tupleImageBounds = (
        min(listNonWhiteCols),  # represents left-most non-white column
        min(listNonWhitekRows),  # represents top-most non-white kRow
        max(listNonWhiteCols) + 1,  # represents right-most non-white column
        max(listNonWhitekRows) + 1   # represents bottom-most non-white kRow
    )
    
    # ASDc: Return tuple containing boundaries.
    return tupleImageBounds

# FUNCTION rotateOptimally(imgIN, deltaAngleIN) ----- ----- ----- ----- -----
# OBJECTIVE:    Find the rotation angle of an image which maximizes its height-to-width ratio.
# INPUT(S):     imgIN: image to rotate;
#               deltaAngleIN: step-size of rotation in degrees;
# OUTPUT(S):    angleOpt: rotation angle at which image height-to-width ratio is maximum;
#               imgRotated: image rotated by angleOpt;

def rotateOptimally(imgIN, deltaAngleIN):

    # ASDc: Initialize ratioMax and angleOpt to zero.
    ratioMax = 0
    angleOpt = 0

    # ASDc: Define myFillColor as white, with length corresponding to number of channels (e.g. 3 for RGB).
    if isinstance(imgIN, (tuple, list)):

        # ASDc: Create tuple with length corresponding to number of channels in pixel (0,0) of imgIN.
        # NOTE: That all values should be 255 (aka. white).
        myFillColor = tuple([255] * len(imgIN.getpixel((0, 0))))

    # ASDc: Repeat this process for black-and-white image with single channel (aka. each pixel is single integer).
    else:
        myFillColor = 255

    # ASDc: Cycle through rotation angles from -45 to +45 degrees, with step-size deltaAngleIN.
    for angleK in range(-90, 90, deltaAngleIN):

        # ASDc: Rotate image by angleK; expand image with white background.
        imgRotatedK = imgIN.rotate(angleK, expand=True, fillcolor=myFillColor)

        # ASDc: Get boundaries of non-white image in imgRotatedK.
        boxRotatedK = getNonWhiteBounds(imgRotatedK)

        # ASDc: Calculate the rotated image's height, width, and ratio (considering non-white regions only).
        heightRotatedK = boxRotatedK[3] - boxRotatedK[1]
        widthRotatedK = boxRotatedK[2] - boxRotatedK[0]
        ratioK = (heightRotatedK / widthRotatedK) if widthRotatedK != 0 else 0

        # ASDc: IF {ratioK is greater than ratioMax} THEN {save current iteration values to maximums}
        if (ratioK > ratioMax) or ((ratioK >= ratioMax) and (abs(angleK) < abs(angleOpt))):
            ratioMax = ratioK
            angleOpt = angleK-deltaAngleIN
        
    # ASDc: Return dictionary output.
    return {
        "angleOpt": angleOpt,
        "imgRotated": imgIN.rotate(angleOpt, expand=True, fillcolor=myFillColor)
    }

# FUNCTION makeImageSquare(imgIN) ----- ----- ----- ----- ----- 
# OBJECTIVE:    Increase size of image (if necessary) to make it square;
# INPUT(S):     imgIN: target image;
# OUTPUT(S):    imgSquareOUT: image made square;

def makeImageSquare(imgIN): 
    
    # ASDc: Get image size in pixels.
    [ widthImg, heightImg ] = imgIN.size

    # ASDc: Determine maximum dimension, which defines length of sequare sides.
    dimMax = max(imgIN.size)

    # ASD: Create new grayscale image of appropriate size.
    imgSquareOUT = Image.new(mode="L", size=(dimMax, dimMax), color=255)
    print("imgSquareOUT size:", imgSquareOUT.size)
    
    imgSquareOUT.show()

    # ASDc: Calculate center/offsetCropped of original imgCropped for copy/pasting.
    # NOTE: That "//" indicates floor division, which ensures that result is an integer.
    offsetCropped = ((dimMax-widthImg) // 2, (dimMax-heightImg) // 2)
    imgSquareOUT.paste(imgIN, offsetCropped)
    
    # ASDc: Return imgSquareOUT.
    return imgSquareOUT

# FUNCTION applyVectorization(imgIN, numSensorsHorizontalIN, numSensorsVerticalIN)
# OBJECTIVE:    Convert image to array of numerical values (aka. convert image to vector);
# INPUT(S):     imgIN: target image;
#               numSensorsHorizontalIN: number of horizotal sensors;
#               numSensorsVerticalIN: number of vertical sensors;
# OUTPUT(S):    arrSensorsOUT: vectorized representation of imgIN;

def applyVectorization(imgIN, numSensorsHorizontalIN, numSensorsVerticalIN):
   
    # ASDc: Get image size in pixels.
    [ widthImg, heightImg ] = imgIN.size  

    # ASDc: Convert image to matrix representation.
    matImgIN = numpy.array(imgIN)    
    
    # ASDc: Create matrix to store boundaries for sensor regions.
    arrHorizontalBounds = numpy.zeros(numSensorsHorizontalIN+1, dtype=int) # NOTE: This must be int as index.
    arrVerticalBounds = numpy.zeros(numSensorsVerticalIN+1, dtype=int) 

    # ASDc: Define boundaries of the horizontal sensors.
    for kCol in range(numSensorsHorizontalIN):
        arrHorizontalBounds[kCol] = (kCol)*widthImg/numSensorsHorizontalIN

    # ASDc: Define boundaries of the vertical sensors.
    for kRow in range(numSensorsVerticalIN):
        arrVerticalBounds[kRow] = (kRow)*heightImg/numSensorsVerticalIN

    # ASDc: Define final horizontal/vertical boundary as width/height of image.
    arrHorizontalBounds[numSensorsHorizontalIN] = widthImg
    arrVerticalBounds[numSensorsVerticalIN] = heightImg
    
    # ASDc: Initialize matrix to store sensor readings.
    matSensorOuts = numpy.zeros((numSensorsVerticalIN, numSensorsHorizontalIN), dtype=numpy.float32)
    
    # ASDc: Cycle through all sensor rows (and columns):
    for kRow in range(numSensorsVerticalIN):
        for kCol in range(numSensorsHorizontalIN):
            
            # ASDc: Calculate the percentage of pixels within this sensor's region that contain a non-white pixel.
            matSensorOuts[kRow, kCol] = 1-(numpy.mean(
                matImgIN[
                    arrVerticalBounds[kRow]:arrVerticalBounds[kRow+1],
                    arrHorizontalBounds[kCol]:arrHorizontalBounds[kCol+1]
                ]
            )/255)

    # ASDc: Convert matSensorOuts from rank-2 matrix to rank-1 array.
    arrSensorsOUT = matSensorOuts.ravel()

    # ASDc: Return arrSensorsOUT.
    return arrSensorsOUT

# FUNCTION genImageVector(pathImgIN, numSensorsHorizontalIN, numSensorsVerticalIN)
# OBJECTIVE:    Accept path for target image and, using the libraries above, generate a vector/numerical representation.
# INPUT(S):     pathImgIN: target image;
#               numSensorsHorizontalIN: number of horizotal sensors;
#               numSensorsVerticalIN: number of vertical sensors;
# OUTPUT(S):    arrImgVectorizedOUT: vectorized representation of imgIN;

def genImageVector(pathImgIN, numSensorsHorizontalIN, numSensorsVerticalIN):

    # ASDc: Employ Image.open() command from PIL library to load image.
    imgPathIN = Image.open(pathImgIN)

    # ASDc: Convert image to RGBA to handle transparency, if needed.
    imgPathIN = imgPathIN.convert("RGBA")

    # ASDc: Convert image to black-and-white.
    imgEdgeBW = convertToBlackAndWhite(imgPathIN)

    # ASDc: Call rotateOptimally() function to get rotated image with maximum height.
    # NOTE: That rotation was eliminated to improve speed.
    #dictRotated = rotateOptimally(imgEdgeBW, 2)

    # ASDc: Call getNonWhiteBounds() function to get boundaries of non-white image.
    boxNonWhiteBounds = getNonWhiteBounds(imgEdgeBW) # NOTE: That rotation was eliminated to improve speed: dictRotated["imgRotated"]

    # ASDc: Crop image according to boxNoneWhiteBounds.
    imgCropped = imgEdgeBW.crop(boxNonWhiteBounds)

    # ASDc: Make image square.
    imgSquareOUT = makeImageSquare(imgCropped)
    
    # ASDc: Apply vectorization.
    arrImgVectorizedOUT = applyVectorization(imgSquareOUT, numSensorsHorizontalIN, numSensorsVerticalIN)

    # ASDc; Return vectorization.
    return arrImgVectorizedOUT