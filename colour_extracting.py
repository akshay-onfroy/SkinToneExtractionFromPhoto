import cv2
import numpy as np

#FINDING THE AVERAGE SKIN TONE BY MASKING THE SKIN REGION AND AVERAGING IT
face = cv2.imread('imgs/test_1.jpg')
cv2.imshow("Face",face)

converted_face = cv2.cvtColor(face,cv2.COLOR_BGR2YCrCb)

lower_range_0 = np.array( [0,133,77] )
upper_range_0 = np.array( [255,190,127] ) 
skin_range_0 = cv2.inRange( converted_face , lower_range_0 ,upper_range_0 )
mask_face = cv2.bitwise_and( converted_face , converted_face , mask=skin_range_0 )

skin_mask_face = mask_face[np.where((mask_face != [0,0,0]).all(axis=2))]

if len(skin_mask_face)>0:
    avg_skin_colour=np.mean(skin_mask_face,axis=0)
    face_avg_skin_colour_ycrcb = np.array(avg_skin_colour)
    face_avg_skin_colour_ycrcb = face_avg_skin_colour_ycrcb.reshape((1,1,3))
    face_avg_skin_colour_bgr = cv2.cvtColor(face_avg_skin_colour_ycrcb.astype(np.uint8),cv2.COLOR_YCrCb2BGR)
    f_b,f_g,f_r = face_avg_skin_colour_bgr[0,0]
    print(f"Average Face Color ",f_b,f_g,f_r)
else:
    print("No skin pixels detected!")

#FINDING THE REGION ON THE MESH WHICH IS TO BE CHANGED THE COLOUR
image = cv2.imread('imgs/bodyMesh.png')
cv2.imshow("Reference Mesh",image)

converted_img = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)

lower_range = np.array( [0,133,77] ) #np.array([0, 133, 77])
upper_range= np.array( [255,190,127] ) #np.array([255, 173, 127])
skin_range = cv2.inRange( converted_img , lower_range,upper_range )
mask = cv2.bitwise_and( converted_img , converted_img , mask=skin_range )

skin_mask = mask[np.where((mask != [0,0,0]).all(axis=2))]

if len(skin_mask)>0:
    avg_skin_colour=np.mean(skin_mask,axis=0)
    avg_skin_colour_ycrcb = np.array(avg_skin_colour)
    avg_skin_colour_ycrcb = avg_skin_colour_ycrcb.reshape((1,1,3))
    avg_skin_colour_bgr = cv2.cvtColor(avg_skin_colour_ycrcb.astype(np.uint8),cv2.COLOR_YCrCb2BGR)
    b,g,r = avg_skin_colour_bgr[0,0]
    print(f"Average Skin Color ",b,g,r)
else:
    print("No skin pixels detected!")

#NOW ONLY CALCULATING THE VALUES OF THE SKIN TONES FOR THE MASKED REGION 
result = image.copy()
skin_pixels = skin_range > 0
result = result.astype(np.float32)
color_ratios = np.array([f_b/b if b != 0 else 1, 
                        f_g/g if g != 0 else 1, 
                        f_r/r if r != 0 else 1], dtype=np.float32)

result[skin_pixels] = np.clip(
    result[skin_pixels] * color_ratios,
    0, 255
)
result = result.astype(np.uint8)

cv2.imshow("Skin Tone Changed Mesh", result)
cv2.imwrite("testing_akshay.png",result)
cv2.waitKey()
cv2.destroyAllWindows()

