import cv2
import numpy as np
import glob,os


def read_in_names(name):

    if name is "hawaii":
        os.chdir("/home/nick/Documents/ComputerVision/hwFour/hawaii/pieces_aligned")
        hawaii_names = glob.glob("*.jpg")
        hawaii_pieces = []
        hawaii_pieces = read_in_pieces(hawaii_names, hawaii_pieces)
        return hawaii_pieces
    if name is "map":
        os.chdir("/home/nick/Documents/ComputerVision/hwFour/map/pieces_aligned")
        map_names = glob.glob("*.jpg")
        
        map_pieces = []
        map_pieces = read_in_pieces(map_names, map_pieces)
       
        return map_pieces

    if name is "trains":
        os.chdir("/home/nick/Documents/ComputerVision/hwFour/trains/pieces_aligned")
        trains_names = glob.glob("*.jpg")
        trains_pieces = []
        trains_pieces = read_in_pieces(trains_names, trains_pieces)
        return trains_pieces



def read_in_pieces(names, map_pieces):

    if len(names) is 0:
        return map_pieces
    else:
        map_pieces.append(cv2.imread(names.pop()))
        return read_in_pieces(names,map_pieces)


def show_stuff(someText, someImage):
    cv2.imshow(someText, someImage)
    cv2.waitKey(0)


def withoutRotation(guideImage, piecesList):

    # SIFT features
    nFeatures = 1000000
    nOctaveLayers = 6
    contrastThreshold = .01  # Threshold to filter out weak features
    edgeThreshold = 15  # Threshold to filter out edges (lower is stricter)
    sigma = 1.6  # The gaussian std dev at octave zero
    
    # Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create(nFeatures, nOctaveLayers, contrastThreshold,
                                       edgeThreshold, sigma)
    
    
    # find the keypoints and descriptors with ORB
    kpGuide, desGuide = sift.detectAndCompute(guideImage,None)
    
    
    
    #empty canvas image to reconstruct picture
    
  
    
    
    # create BFMatcher object
    bf = cv2.BFMatcher()
    
   
    bad = []

    for pieces in piecesList :
        kpSample, desSample = sift.detectAndCompute(pieces,None)
        # Match descriptors.
        if desSample is None:
              
                bad.append(pieces)
                continue
        matches = bf.match( desGuide,desSample)
        #matches = bf.knnMatch(desGuide, desSample)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
    
        
        
        for match in matches[:2]:
    
            curr_kp1 = kpGuide[match.queryIdx]  # get the keypoint for img1
            loc1 = curr_kp1.pt
            
            
            x1 = int(loc1[0])
            y1 = int(loc1[1])
            x1 = (x1//50)*50
            y1 = (y1//50)*50
           
           
            canvas[y1:y1 + 50, x1:x1 + 50] = pieces
        show_stuff("as it loops", canvas)



map_pieces = read_in_names("map")
hawaii_pieces = read_in_names("hawaii")

guideImage = cv2.imread('/home/nick/Documents/ComputerVision/hwFour/hawaii/hawaii_full.jpg')
show_stuff("original", guideImage)
rows, cols = guideImage.shape[:2]
canvas = np.zeros((rows, cols,3), np.uint8)
withoutRotation(guideImage, hawaii_pieces)

show_stuff("canvas finished", canvas)





