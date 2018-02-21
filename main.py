import cv2
import dlib
from skimage import feature
from skimage import exposure

# load the source picture
color_image = cv2.imread('src/picture_1.png')

# save the source picture
cv2.imwrite('output/1.png', color_image)

# convert to grayscale
image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# save the grayscale image
cv2.imwrite('output/2.png', image)

# get visualization of the histogram of oriented gradients
(H, hog_image) = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
hog_image = exposure.rescale_intensity(hog_image, out_range=(0, 255))
hog_image = hog_image.astype('uint8')

# save the visualization of the histogram of oriented gradients
cv2.imwrite('output/3.png', hog_image)

# load the required pre-trained face detection model
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "shape_predictor_68_face_landmarks.dat"

# create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

# run the HOG face detector on the image data
detected_faces = face_detector(image, 1)

# draw red squares around the faces
for face in detected_faces:
    cv2.rectangle(hog_image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

# save the copy
cv2.imwrite('output/4.png', hog_image)

# show the image in a desktop window
win = dlib.image_window()
win.set_image(image)

# loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):
    # draw a box around each face we found
    win.add_overlay(face_rect)

    # get the the face's pose
    pose_landmarks = face_pose_predictor(image, face_rect)

    # draw the face landmarks on the screen
    win.add_overlay(pose_landmarks)

dlib.hit_enter_to_continue()
