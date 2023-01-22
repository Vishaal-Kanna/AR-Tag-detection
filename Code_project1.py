#!/usr/bin/env python

"""
ENPM673: Perception for Autonomous Robots
Projec 1

Vishaal Kanna Sivakumar (vishaal@terpmail.umd.edu)
M.Eng. Student, Robotics
University of Maryland, College Park

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def AR_tag_operations(file_name):
	vid_capture = cv2.VideoCapture(file_name)
	k=0

	while (vid_capture.isOpened()):
		ret, frame = vid_capture.read()
		if ret == True:
			if k%10==0:
				img = frame

				#image converted to grayscale
				imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

				#Applying Fourier Transform
				dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(imgg))
				plt.imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
				f = np.log(abs(dark_image_grey_fourier))
				r=100
				cx = 960
				cy= 540

				#Creating a mask with radius 100px
				for i in range(int(f.shape[0]/2)-r,int(f.shape[0]/2)+r):
					for j in range(int(f.shape[1]/2) - r, int(f.shape[1]/2) + r):
						if ((i-cy)**2+(j-cx)**2)**0.5<=r:
							dark_image_grey_fourier[i][j]=0

				#Taking Inverse FT and thresholding to get edges
				edges = abs(np.fft.ifft2(dark_image_grey_fourier))
				edges1 = 0*edges
				edges1[edges>=0.5*edges.max()]=255

				#Filtering the edge points to obtain candidates for Corners using a perturbation of 30px followed by 20px
				lst=[]
				p=30
				for i in range(p,imgg.shape[0]-p-1):
					for j in range(p, imgg.shape[1]-p-1):
						count = 0
						if edges1[i][j]==255:
							if imgg[i+p][j] >=150:
								count=count+1
							if imgg[i][j+p] >=150:
								count=count+1
							if imgg[i-p][j] >=150:
								count=count+1
							if imgg[i][j-p] >=150:
								count=count+1
							if count==3:
								lst.append((i,j))

				p=20
				imgg1=0*imgg
				imgg1[imgg>=100]=255
				lst2=[]
				for i in range(0,len(lst)):
					count=0
					if imgg1[lst[i][0] + p][lst[i][1]+p] == 0:
						count = count + 1
					if imgg1[lst[i][0] + p][lst[i][1] - p] == 0:
						count = count + 1
					if imgg1[lst[i][0] - p][lst[i][1] + p] == 0:
						count = count + 1
					if imgg1[lst[i][0] - p][lst[i][1] - p] == 0:
						count = count + 1
					if count == 1:
						lst2.append((lst[i][0],lst[i][1]))

				#Algorithm to obtain 4 corner points from the candidates filtered from previous steps
				points_uns = []
				maxA = 0
				maxD1 = 0
				maxD2 = 0
				for i in range(0, 10000):
					p1 = np.random.randint(0, len(lst2))
					p2 = np.random.randint(0, len(lst2))
					p3 = np.random.randint(0, len(lst2))
					p4 = np.random.randint(0, len(lst2))
					if p1 == p2 or p1 == p3 or p1 == p4 or p2 == p3 or p2 == p4 or p3 == p4:
						continue
					x1 = lst2[p1][0]
					x2 = lst2[p2][0]
					x3 = lst2[p3][0]
					x4 = lst2[p4][0]
					y1 = lst2[p1][1]
					y2 = lst2[p2][1]
					y3 = lst2[p3][1]
					y4 = lst2[p4][1]
					if ((((x2+x3)/2)-((x1+x4)/2))**2+(((y2+y3)/2)-((y1+y4)/2))**2)**5>=10:
						continue
					if ((x1-x4)**2+(y1-y4)**2)**0.5>=100 and ((x3-x2)**2+(y3-y2)**2)**0.5>=100 and ((x1-x2)**2+(y1-y2)**2)**0.5>=100 and ((x3-x4)**2+(y3-y4)**2)**0.5>=100 and ((x3-x1)**2+(y3-y1)**2)**0.5>=100 and ((x2-x4)**2+(y2-y4)**2)**0.5>=100:
						A1 = 0.5 * abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))
						A2 = 0.5 * abs((x4 * (y2 - y3) + x2 * (y3 - y4) + x3 * (y4 - y2)))
						if ((x3-x2)**2+(y3-y2)**2)**0.5 >maxD1 and ((x1-x4)**2+(y1-y4)**2)**0.5 >maxD2:
							if A1 + A2 > maxA:
								points_uns.append(p1)
								points_uns.append(p3)
								points_uns.append(p4)
								points_uns.append(p2)
								maxA = A1 + A2
								maxD1 = ((x3-x2)**2+(y3-y2)**2)**0.5
								maxD2 = ((x1-x4)**2+(y1-y4)**2)**0.5

				if len(points_uns)<4:
					cv2.imshow("Cube", img)
					cv2.waitKey(10)
					continue
				if len(lst2)<max(points_uns)-1:
					cv2.imshow("Cube", img)
					cv2.waitKey(10)
					continue

				BL, BR = -1,-1

				#Detecting the bottom right corner and sorting them to be in anti-clockwise direction
				x1 = lst2[points_uns[0]][1]
				y1 = lst2[points_uns[0]][0]
				x2 = lst2[points_uns[2]][1]
				y2 = lst2[points_uns[2]][0]
				d = 3*(((x2-x1)**2+(y2-y1)**2)**0.5)/8
				x = int((5.5*x1+2.5*x2)/8)
				y = int((5.5*y1+2.5*y2)/8)
				xb = int((4.5 * x1 + 3.5 * x2) / 8)
				yb = int((4.5 * y1 + 3.5 * y2) / 8)
				if imgg[y][x] >=150:
					BR = 0
				if imgg[yb][xb] <=50:
					BL = 0

				x1 = lst2[points_uns[2]][1]
				y1 = lst2[points_uns[2]][0]
				x2 = lst2[points_uns[0]][1]
				y2 = lst2[points_uns[0]][0]
				x = int((5.5*x1+2.5*x2)/8)
				y = int((5.5*y1+2.5*y2)/8)
				xb = int((4.5 * x1 + 3.5 * x2) / 8)
				yb = int((4.5 * y1 + 3.5 * y2) / 8)
				if imgg[y][x] >=150:
					BR = 2
				if imgg[yb][xb] <=50:
					BL = 2

				x1 = lst2[points_uns[1]][1]
				y1 = lst2[points_uns[1]][0]
				x2 = lst2[points_uns[3]][1]
				y2 = lst2[points_uns[3]][0]
				d = 3*(((x2-x1)**2+(y2-y1)**2)**0.5)/8
				x = int((5.5*x1+2.5*x2)/8)
				y = int((5.5*y1+2.5*y2)/8)
				xb = int((4.5 * x1 + 3.5 * x2) / 8)
				yb = int((4.5 * y1 + 3.5 * y2) / 8)
				if imgg[y][x] >=150:
					BR = 1
				if imgg[yb][xb] <=50:
					BL = 1

				x1 = lst2[points_uns[3]][1]
				y1 = lst2[points_uns[3]][0]
				x2 = lst2[points_uns[1]][1]
				y2 = lst2[points_uns[1]][0]
				d = 3*(((x2-x1)**2+(y2-y1)**2)**0.5)/8
				x = int((5.5*x1+2.5*x2)/8)
				y = int((5.5*y1+2.5*y2)/8)
				xb = int((4.5 * x1 + 3.5 * x2) / 8)
				yb = int((4.5 * y1 + 3.5 * y2) / 8)
				if imgg[y][x] >=150:
					BR = 3
				if imgg[yb][xb] <=50:
					BL = 3

				if BR==-1 or BL==-1:
					cv2.imshow("Cube", img)
					cv2.waitKey(10)
					continue

				points=[]

				if BL-BR==1:
					points.append(points_uns[BR])
					points.append(points_uns[BL])
					if BR == 0:
						points.append(points_uns[BL+1])
						points.append(points_uns[BL+2])
					if BR == 1:
						points.append(points_uns[BL+1])
						points.append(points_uns[BR-1])
					if BR == 2:
						points.append(points_uns[BR-2])
						points.append(points_uns[BR-1])

				if BR==3:
					if BL ==0:
						points.append(points_uns[BR])
						points.append(points_uns[BL])
						points.append(points_uns[BL+1])
						points.append(points_uns[BL+2])

				if BR-BL==1:
					points.append(points_uns[BR])
					points.append(points_uns[BL])
					if BR == 1:
						points.append(points_uns[BR+2])
						points.append(points_uns[BR+1])
					if BR == 2:
						points.append(points_uns[BR-1])
						points.append(points_uns[BR+1])
					if BR == 3:
						points.append(points_uns[BR-1])
						points.append(points_uns[BR-2])

				if BR==0:
					if BL ==3:
						points.append(points_uns[BR])
						points.append(points_uns[BL])
						points.append(points_uns[BL-1])
						points.append(points_uns[BL-2])

				#Homography estimation using the reference image of 200x200px
				tag_ref = cv2.imread("ref_marker.png", 0)
				testudo = cv2.imread("testudo.png")
				imgH = tag_ref.shape[0]
				imgW = tag_ref.shape[1]

				if len(points)<4:
					cv2.imshow("Cube", img)
					cv2.waitKey(10)
					continue

				x1 = lst2[points[0]][1]
				x2 = lst2[points[1]][1]
				x3 = lst2[points[2]][1]
				x4 = lst2[points[3]][1]
				y1 = lst2[points[0]][0]
				y2 = lst2[points[1]][0]
				y3 = lst2[points[2]][0]
				y4 = lst2[points[3]][0]

				xp1 = 0
				xp2 = imgW - 1
				xp3 = imgH-1
				xp4 = 0
				yp1 = imgW
				yp2 = imgH - 1
				yp3 = 0
				yp4 = 0

				#Using SVD for solving the AH=0 equation
				A = np.matrix([[-x1, -y1, -1, 0, 0, 0, x1 * xp1, y1 * xp1, xp1],
							   [0, 0, 0, -x1, -y1, -1, x1 * yp1, y1 * yp1, yp1],
							   [-x2, -y2, -1, 0, 0, 0, x2 * xp2, y2 * xp2, xp2],
							   [0, 0, 0, -x2, -y2, -1, x2 * yp2, y2 * yp2, yp2],
							   [-x3, -y3, -1, 0, 0, 0, x3 * xp3, y3 * xp3, xp3],
							   [0, 0, 0, -x3, -y3, -1, x3 * yp3, y3 * yp3, yp3],
							   [-x4, -y4, -1, 0, 0, 0, x4 * xp4, y4 * xp4, xp4],
							   [0, 0, 0, -x4, -y4, -1, x4 * yp4, y4 * yp4, yp4]])

				U,S,V = np.linalg.svd(A, full_matrices=True)
				H = (V[8, :] / V[8, 8]).reshape(3, 3)
				H_inv = np.linalg.inv(H)

				#Warping the AR-Tag to the reference tag plane for decoding
				warped_img = np.zeros([tag_ref.shape[0], tag_ref.shape[1]])
				for m in range(tag_ref.shape[0]):
					for n in range(tag_ref.shape[1]):
						mat = np.matrix([m,n,1])
						prod = H_inv*mat.T
						h1 = prod[0]
						w1 = prod[1]
						z1 = prod[2]
						ha, wa = int(h1 / z1), int(w1 / z1)
						if (ha < imgg.shape[1] and ha > 0 and wa < imgg.shape[0] and wa > 0):
							warped_img[m][n] = imgg[wa][ha]

				threshold = 100
				AR_tag = np.zeros((4,4))
				w_cell = int(tag_ref.shape[0]/8)
				for i in range(2,6):
					for j in range(2, 6):
						if warped_img[int((i+0.5)*w_cell)][int((j+0.5)*w_cell)]>=threshold:
							AR_tag[i-2][j-2]=1
				rot=0
				for i in range(0,4):
					AR_tag = np.rot90(AR_tag)
					if AR_tag[3][3]==1:
						rot=i+1
						break

				#Tag ID calculation after decoding the AR-tag
				Tag_id = 8*AR_tag[1][1]+4*AR_tag[1][2]+2*AR_tag[2][2]+1*AR_tag[2][1]
				print(Tag_id)

				#Projection matrix Calculation from K and H_inv
				K = np.array([[1346.100595,	0, 932.1633975],[0, 1355.933136, 654.8986796],[0, 0, 1]])
				inv_K = np.linalg.inv(K)
				B = np.matmul(inv_K, H_inv)
				b1 = B[:, 0].reshape(3, 1)
				b2 = B[:, 1].reshape(3, 1)
				b3 = B[:, 2].reshape(3, 1)

				h1 = H_inv[:, 0].reshape(3, 1)
				h2 = H_inv[:, 1].reshape(3, 1)
				lamb_da = 2 / (np.linalg.norm(inv_K.dot(h1)) + np.linalg.norm(inv_K.dot(h2)))
				t = lamb_da * b3
				r1 = lamb_da * b1
				r2 = lamb_da * b2
				r3 = (np.cross(r1.reshape(-1), r2.reshape(-1))).reshape(3, 1)
				RT = np.concatenate((r1, r2, r3, t), axis=1)
				P = np.matmul(K, RT)

				# Warping the testudo image onto the image after rotation
				for i in range(0,4-rot):
					testudo = np.rot90(testudo)

				testudo = cv2.resize(testudo, (tag_ref.shape[1], tag_ref.shape[0]))
				img1 = img
				for m in range(testudo.shape[0]):
					for n in range(testudo.shape[1]):
						mat = np.matrix([m, n, 0, 1])
						prod = P * mat.T
						h1 = prod[0]
						w1 = prod[1]
						z1 = prod[2]
						ha, wa = int(h1 / z1), int(w1 / z1)
						if (ha < imgg.shape[1] and ha > 0 and wa < imgg.shape[0] and wa > 0):
							img1[wa][ha][0] = testudo[m][n][0]
							img1[wa][ha][1] = testudo[m][n][1]
							img1[wa][ha][2] = testudo[m][n][2]

				#Defining the world coordinates and calculating the projected coordinates in the image
				cube_w = np.matrix([[0, 0, 0, 1], [imgH-1, 0, 0, 1], [imgH-1, imgW-1, 0, 1],[0, imgW-1, 0, 1],[0, 0, 200, 1], [imgH-1, 0, 200, 1], [imgH-1, imgW-1, 200, 1], [0, imgW-1, 200, 1]])
				cube_img = []

				for i in range(0, 8):
					prod = np.matmul(P, cube_w[i].T)
					h1 = prod[0]
					w1 = prod[1]
					z1 = prod[2]
					ha, wa = int(h1 / z1), int(w1 / z1)
					if (ha < imgg.shape[1] and ha > 0 and wa < imgg.shape[0] and wa > 0):
						cube_img.append((ha,wa))

				if len(cube_img)<8:
					cv2.imshow("Cube", img)
					cv2.waitKey(10)
					continue

				#Connecting the corners to form a cube
				img = cv2.line(img, cube_img[0], cube_img[1], [0, 0, 255], 2)
				img = cv2.line(img, cube_img[1], cube_img[2], [0, 0, 255], 2)
				img = cv2.line(img, cube_img[2], cube_img[3], [0, 0, 255], 2)
				img = cv2.line(img, cube_img[3], cube_img[0], [0, 0, 255], 2)

				img = cv2.line(img, cube_img[4], cube_img[5], [0, 0, 255], 2)
				img = cv2.line(img, cube_img[5], cube_img[6], [0, 0, 255], 2)
				img = cv2.line(img, cube_img[6], cube_img[7], [0, 0, 255], 2)
				img = cv2.line(img, cube_img[7], cube_img[4], [0, 0, 255], 2)

				img = cv2.line(img, cube_img[0], cube_img[4], [0, 0, 255], 2)
				img = cv2.line(img, cube_img[1], cube_img[5], [0, 0, 255], 2)
				img = cv2.line(img, cube_img[2], cube_img[6], [0, 0, 255], 2)
				img = cv2.line(img, cube_img[3], cube_img[7], [0, 0, 255], 2)

				cv2.imshow("Cube", img)
				cv2.waitKey(100)

			k=k+1
		else:
			break

def main():
	AR_tag_operations('1tagvideo.mp4')

if __name__ == '__main__':
	main()


