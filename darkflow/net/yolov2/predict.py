import numpy as np
import math
import cv2
import os
import json
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor

import shutil


def expit(x):
	return 1. / (1. + np.exp(-x))


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()
	boxes=box_constructor(meta, net_out)
	return boxes


# 블러링 처리해서 이미지 반환
def bluring(imgcv, coord):
	# 원본 이미지 복사 후 medianBlur 처리
	dst = imgcv.copy()
	blurred = cv2.medianBlur(dst, 25)
	# blurred = cv2.blur(dst, (15, 15))
	# blurred = cv2.GaussianBlur(dst, (5, 5), 0)

	# RGB 코드
	white_color = (255, 255, 255)
	black_color = (0, 0, 0)
	red_color = (0, 0, 255)

	# 검정/흰색 배경 마스크 생성
	mask = np.zeros(dst.shape[:2], np.uint8)
	mask_inv = np.zeros(dst.shape[:2], np.uint8)
	mask_inv.fill(255)

	# 내접 원형 생성 (블러링 지점)
	for i in coord:
		center = (int((i[2]+i[0])/2), int((i[1]+i[3])/2))
		size = (int((i[2]-i[0])/4), int((i[3]-i[1])/4))

		cv2.ellipse(mask, center, size, 0, 0, 360, white_color, -1)
		cv2.ellipse(mask_inv, center, size, 0, 0, 360, black_color, -1)

	# 마스크 적용 + 비트 연산을 통한 합성
	img_bg = cv2.bitwise_and(dst, dst, mask=mask_inv)
	img_fg = cv2.bitwise_and(blurred, blurred, mask=mask)
	dst = cv2.add(img_bg, img_fg)

	# 데모를 위한 블러링 위치 생성
	demo = imgcv.copy()
	for i in coord:
		center = (int((i[2] + i[0]) / 2), int((i[1] + i[3]) / 2))
		size = (int((i[2] - i[0]) / 4), int((i[3] - i[1]) / 4))

		cv2.ellipse(demo, center, size, 0, 0, 360, red_color, 1)

	return dst, demo

#
# 경식이가 추가한 처리
#
def Rtt(height, width, degree, img):
	matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
	img = cv2.warpAffine(img, matrix, (width, height))
	img = img[1:width - 1, 1:height - 1]
	img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
	return img

def Bluring(img, blr, height, width):
    img = cv2.resize(img, (blr, blr), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return img

def Cnct(img, y1, y2, x1, x2):
	line = img[y1:y2, x1:x2][:]
	line2 = cv2.filter2D(line, -1, np.ones((5, 5), np.float32) / 25.0)
	img[y1:y2, x1:x2] = line2
	return img


detection_choice = 0

def postprocess(self, net_out, im, save = True):
	# 지문
	if detection_choice == 1:
		"""
		Takes net output, draw net_out, save to disk
		"""
		boxes = self.findboxes(net_out)

		# meta
		meta = self.meta
		threshold = meta['thresh']
		colors = meta['colors']
		labels = meta['labels']
		if type(im) is not np.ndarray:
			imgcv = cv2.imread(im)
		else:
			imgcv = im
		h, w, _ = imgcv.shape

		# 데모용 카피 이미지
		copyimg = imgcv.copy()

		resultsForJSON = []
		resultArray = []
		# 이미지의 좌표 생성
		resultCoord = []
		for b in boxes:
			boxResults = self.process_box(b, h, w, threshold)
			if boxResults is None:
				continue
			left, right, top, bot, mess, max_indx, confidence = boxResults
			thick = int((h + w) // 400) # 300
			if self.FLAGS.json:
				resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
				continue
			if self.FLAGS.array:
				resultArray.append([left, top, right, bot])
			resultCoord.append([left, top, right, bot])

			cv2.rectangle(imgcv, (left, top), (right, bot), colors[max_indx], thick)
			# cv2.putText(imgcv, mess, (left, top - 12), 0, 1e-3 * h, colors[max_indx], thick//3)
			# cv2.putText(imgcv, '', (left, top - 12), 0, 1e-3 * h, colors[max_indx], thick // 3)

		if not save:
			for a in resultArray:
				print(a)

			return imgcv

		outfolder = os.path.join(self.FLAGS.imgdir, 'out')
		img_name = os.path.join(outfolder, os.path.basename(im))

		# 복사를 위한
		copy_src_path = img_name
		copy_dst_path = self.FLAGS.imgdir # os.path.join(self.FLAGS.imgdir, os.path.basename(im))

		print('복사 SRC: ' + copy_src_path)
		print('복사 DST: ' + copy_dst_path)

		if self.FLAGS.json:
			textJSON = json.dumps(resultsForJSON)
			textFile = os.path.splitext(img_name)[0] + ".json"
			with open(textFile, 'w') as f:
				f.write(textJSON)
			return
		if self.FLAGS.array:
			textFile = os.path.splitext(img_name)[0] + ".txt"
			with open(textFile, 'w') as f:
				result = ''
				for a in resultArray:
					print(a)
					for b in a:
						result = result + str(b) + ','
					result = result.rstrip(',')
					result += '\n'
				f.write(result)

		# 오브젝트 디텍션 결과 출력
		# Coordinates = [left, top, right, bottom]
		print('Object Detection Result')
		print('Coord: ', resultCoord)

		# 디텍션 결과 저장
		cv2.imwrite(os.path.splitext(img_name)[0] + '_1_detect.jpg', imgcv)
		# 이미지 블러링 처리
		dst = bluring(copyimg, resultCoord)

		# opyimg = cv2.blur(copyimg, (15, 15))

		# cv2.imwrite(os.path.splitext(img_name)[0] + '_temp.jpg', dst)
		cv2.imwrite(img_name, dst[0])
		# cv2.imwrite(os.path.splitext(img_name)[0] + '_3_dst.jpg', dst[0])
		cv2.imwrite(os.path.splitext(img_name)[0] + '_2_blur.jpg', dst[1])
		cv2.imwrite(os.path.splitext(img_name)[0] + '_4_comp.jpg', copyimg)

		# 원본 쪽에 복사
		shutil.copy(copy_src_path, copy_dst_path)
	#
	# 홍채 : 경식
	#
	elif detection_choice == 2:
		"""
			Takes net output, draw net_out, save to disk
			"""
		boxes = self.findboxes(net_out)

		# meta
		meta = self.meta
		threshold = meta['thresh']
		colors = meta['colors']
		labels = meta['labels']
		if type(im) is not np.ndarray:
			imgcv = cv2.imread(im)
		else:
			imgcv = im
		h, w, _ = imgcv.shape

		# 박스 치는 곳
		resultCoord = []  # 디텍션 좌표 넣어둘 리스트
		resultsForJSON = []
		for b in boxes:
			boxResults = self.process_box(b, h, w, threshold)
			if boxResults is None:
				continue
			left, right, top, bot, mess, max_indx, confidence = boxResults
			thick = int((h + w) // 300)
			if self.FLAGS.json:
				resultsForJSON.append(
					{"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top},
					 "bottomright": {"x": right, "y": bot}})
				continue

			# 리스트에 좌표 등록
			resultCoord.append([left, top, right, bot])

			'''
            cv2.rectangle(imgcv,
                (left, top), (right, bot),
                colors[max_indx], thick)
            cv2.putText(imgcv, mess, (left, top - 12),
                0, 1e-3 * h, colors[max_indx],thick//3)
            '''

		if not save: return imgcv

		outfolder = os.path.join(self.FLAGS.imgdir, 'out')
		img_name = os.path.join(outfolder, os.path.basename(im))
		if self.FLAGS.json:
			textJSON = json.dumps(resultsForJSON)
			textFile = os.path.splitext(img_name)[0] + ".json"
			with open(textFile, 'w') as f:
				f.write(textJSON)
			return

		# print('디텍션 결과')
		# print('박스좌표: ', resultCoord)

		dst_img = []
		for i in resultCoord:
			dst = imgcv.copy()
			dst_img.append(dst[i[1]:i[3], i[0]:i[2]])

		height, width, channel = imgcv.shape
		tempImg = np.zeros((height, width, 3), np.uint8)

		for i in resultCoord:
			x_offset1 = i[0]
			x_offset2 = i[2]
			y_offset1 = i[1]
			y_offset2 = i[3]

			pred_wt = x_offset2 - x_offset1
			pred_ht = y_offset2 - y_offset1

			iris = imgcv[y_offset1:y_offset2, x_offset1:x_offset2]

			if pred_ht > 200:
				resized = cv2.resize(iris, (200, 200), interpolation=cv2.INTER_AREA)
			else:
				resized = cv2.resize(iris, (200, 200), interpolation=cv2.INTER_CUBIC)

			#######################  구역 설정   #################

			co_a, co_b, co_c, co_d, co_e, co_f = 32, 40, 48, 56, 64, 72
			co_g, co_h, co_i, co_j, co_k, co_l = 80, 88, 96, 104, 112, 120
			co_m, co_n, co_o, co_p, co_q, co_r = 128, 136, 144, 152, 160, 168

			ht = wt = co_b - co_a
			bl = 6

			div00 = resized[co_a:co_b, co_f:co_g][:]
			div00 = Bluring(div00, bl, ht, wt)
			div00 = Rtt(ht, wt, 90, div00)

			div01 = resized[co_a:co_b, co_g:co_h][:]
			div01 = Bluring(div01, bl, ht, wt)
			div01 = Rtt(ht, wt, -90, div01)

			div02 = resized[co_a:co_b, co_h:co_i][:]
			div02 = Bluring(div02, bl, ht, wt)
			div02 = Rtt(ht, wt, 90, div02)

			div03 = resized[co_a:co_b, co_i:co_j][:]
			div03 = Bluring(div03, bl, ht, wt)
			div03 = Rtt(ht, wt, 180, div03)

			div04 = resized[co_a:co_b, co_j:co_k][:]
			div04 = Bluring(div04, bl, ht, wt)
			div04 = Rtt(ht, wt, -90, div04)

			div05 = resized[co_a:co_b, co_k:co_l][:]
			div05 = Bluring(div05, bl, ht, wt)

			div06 = resized[co_a:co_b, co_l:co_m][:]
			div06 = Bluring(div06, bl, ht, wt)
			div06 = Rtt(ht, wt, 90, div05)

			div07 = resized[co_b:co_c, co_d:co_e][:]
			div07 = Bluring(div07, bl, ht, wt)
			div07 = Rtt(ht, wt, 90, div06)

			div08 = resized[co_b:co_c, co_e:co_f][:]
			div08 = Bluring(div08, bl, ht, wt)
			div08 = Rtt(ht, wt, 180, div08)

			div09 = resized[co_b:co_c, co_f:co_g][:]
			div09 = Bluring(div09, bl, ht, wt)
			div09 = Rtt(ht, wt, -90, div09)

			div10 = resized[co_b:co_c, co_g:co_h][:]
			div10 = Bluring(div10, bl, ht, wt)
			div10 = Rtt(ht, wt, 90, div10)

			div11 = resized[co_b:co_c, co_h:co_i][:]
			div11 = Bluring(div11, bl, ht, wt)
			div11 = Rtt(ht, wt, 180, div11)

			div12 = resized[co_b:co_c, co_i:co_j][:]
			div12 = Bluring(div12, bl, ht, wt)
			div12 = Rtt(ht, wt, 180, div12)

			div13 = resized[co_b:co_c, co_j:co_k][:]
			div13 = Bluring(div13, bl, ht, wt)
			div13 = Rtt(ht, wt, 90, div13)

			div14 = resized[co_b:co_c, co_k:co_l][:]
			div14 = Bluring(div14, bl, ht, wt)
			div14 = Rtt(ht, wt, 180, div14)

			div15 = resized[co_b:co_c, co_l:co_m][:]
			div15 = Bluring(div15, bl, ht, wt)
			div15 = Rtt(ht, wt, -90, div15)

			div16 = resized[co_b:co_c, co_m:co_n][:]
			div16 = Bluring(div16, bl, ht, wt)

			div17 = resized[co_b:co_c, co_n:co_o][:]
			div17 = Bluring(div17, bl, ht, wt)
			div17 = Rtt(ht, wt, 90, div17)

			div18 = resized[co_c:co_d, co_c:co_d][:]
			div18 = Bluring(div18, bl, ht, wt)
			div18 = Rtt(ht, wt, 180, div18)

			div19 = resized[co_c:co_d, co_d:co_e][:]
			div19 = Bluring(div19, bl, ht, wt)
			div19 = Rtt(ht, wt, -90, div19)

			div20 = resized[co_c:co_d, co_e:co_f][:]
			div20 = Bluring(div20, bl, ht, wt)

			div21 = resized[co_c:co_d, co_m:co_n][:]
			div21 = Bluring(div21, bl, ht, wt)
			div21 = Rtt(ht, wt, -90, div21)

			div22 = resized[co_c:co_d, co_n:co_o][:]
			div22 = Bluring(div22, bl, ht, wt)
			div22 = Rtt(ht, wt, 90, div22)

			div23 = resized[co_c:co_d, co_o:co_p][:]
			div23 = Bluring(div23, bl, ht, wt)
			div23 = Rtt(ht, wt, 90, div23)

			div24 = resized[co_d:co_e, co_b:co_c][:]
			div24 = Bluring(div24, bl, ht, wt)
			div24 = Rtt(ht, wt, 180, div24)

			div25 = resized[co_d:co_e, co_c:co_d][:]
			div25 = Bluring(div25, bl, ht, wt)
			div25 = Rtt(ht, wt, -90, div25)

			div26 = resized[co_d:co_e, co_d:co_e][:]
			div26 = Bluring(div26, bl, ht, wt)
			div26 = Rtt(ht, wt, 180, div26)

			div27 = resized[co_d:co_e, co_n:co_o][:]
			div27 = Bluring(div27, bl, ht, wt)
			div27 = Rtt(ht, wt, 90, div27)

			div28 = resized[co_d:co_e, co_o:co_p][:]
			div28 = Bluring(div28, bl, ht, wt)
			div28 = Rtt(ht, wt, -90, div28)

			div29 = resized[co_d:co_e, co_p:co_q][:]
			div29 = Bluring(div29, bl, ht, wt)
			div29 = Rtt(ht, wt, -90, div29)

			div30 = resized[co_e:co_f, co_b:co_c][:]
			div30 = Bluring(div30, bl, ht, wt)

			div31 = resized[co_e:co_f, co_c:co_d][:]
			div31 = Bluring(div31, bl, ht, wt)
			div31 = Rtt(ht, wt, 90, div31)

			div32 = resized[co_e:co_f, co_o:co_p][:]
			div32 = Bluring(div32, bl, ht, wt)
			div32 = Rtt(ht, wt, 180, div32)

			div33 = resized[co_e:co_f, co_p:co_q][:]
			div33 = Bluring(div33, bl, ht, wt)
			div33 = Rtt(ht, wt, 180, div33)

			div34 = resized[co_f:co_g, co_a:co_b][:]
			div34 = Bluring(div34, bl, ht, wt)
			div34 = Rtt(ht, wt, 90, div34)

			div35 = resized[co_f:co_g, co_b:co_c][:]
			div35 = Bluring(div35, bl, ht, wt)
			div35 = Rtt(ht, wt, -90, div35)

			div36 = resized[co_f:co_g, co_p:co_q][:]
			div36 = Bluring(div36, bl, ht, wt)
			div36 = Rtt(ht, wt, 90, div36)

			div37 = resized[co_f:co_g, co_q:co_r][:]
			div37 = Bluring(div37, bl, ht, wt)
			div37 = Rtt(ht, wt, -90, div37)

			div38 = resized[co_g:co_h, co_a:co_b][:]
			div38 = Bluring(div38, bl, ht, wt)
			div38 = Rtt(ht, wt, 180, div38)

			div39 = resized[co_g:co_h, co_b:co_c][:]
			div39 = Bluring(div39, bl, ht, wt)

			div40 = resized[co_g:co_h, co_p:co_q][:]
			div40 = Bluring(div40, bl, ht, wt)
			div40 = Rtt(ht, wt, 180, div40)

			div41 = resized[co_g:co_h, co_q:co_r][:]
			div41 = Bluring(div41, bl, ht, wt)
			div41 = Rtt(ht, wt, 90, div41)

			div42 = resized[co_h:co_i, co_a:co_b][:]
			div42 = Bluring(div42, bl, ht, wt)
			div42 = Rtt(ht, wt, 180, div42)

			div43 = resized[co_h:co_i, co_b:co_c][:]
			div43 = Bluring(div43, bl, ht, wt)
			div43 = Rtt(ht, wt, 90, div43)

			div44 = resized[co_h:co_i, co_p:co_q][:]
			div44 = Bluring(div44, bl, ht, wt)

			div45 = resized[co_h:co_i, co_q:co_r][:]
			div45 = Bluring(div45, bl, ht, wt)
			div45 = Rtt(ht, wt, -90, div45)

			div46 = resized[co_i:co_j, co_a:co_b][:]
			div46 = Bluring(div46, bl, ht, wt)
			div46 = Rtt(ht, wt, -90, div46)

			div47 = resized[co_i:co_j, co_b:co_c][:]
			div47 = Bluring(div47, bl, ht, wt)

			div48 = resized[co_i:co_j, co_p:co_q][:]
			div48 = Bluring(div48, bl, ht, wt)
			div48 = Rtt(ht, wt, 90, div48)

			div49 = resized[co_i:co_j, co_q:co_r][:]
			div49 = Bluring(div49, bl, ht, wt)
			div49 = Rtt(ht, wt, 90, div49)

			div50 = resized[co_j:co_k, co_a:co_b][:]
			div50 = Bluring(div50, bl, ht, wt)
			div50 = Rtt(ht, wt, 180, div50)

			div51 = resized[co_j:co_k, co_b:co_c][:]
			div51 = Bluring(div51, bl, ht, wt)
			div51 = Rtt(ht, wt, 180, div51)

			div52 = resized[co_j:co_k, co_p:co_q][:]
			div52 = Bluring(div52, bl, ht, wt)
			div52 = Rtt(ht, wt, -90, div52)

			div53 = resized[co_j:co_k, co_q:co_r][:]
			div53 = Bluring(div53, bl, ht, wt)
			div53 = Rtt(ht, wt, 90, div53)

			div54 = resized[co_k:co_l, co_a:co_b][:]
			div54 = Bluring(div54, bl, ht, wt)
			div54 = Rtt(ht, wt, 180, div54)

			div55 = resized[co_k:co_l, co_b:co_c][:]
			div55 = Bluring(div55, bl, ht, wt)
			div55 = Rtt(ht, wt, 180, div55)

			div56 = resized[co_k:co_l, co_p:co_q][:]
			div56 = Bluring(div56, bl, ht, wt)
			div56 = Rtt(ht, wt, 180, div56)

			div57 = resized[co_k:co_l, co_q:co_r][:]
			div57 = Bluring(div57, bl, ht, wt)

			div58 = resized[co_l:co_m, co_a:co_b][:]
			div58 = Bluring(div58, bl, ht, wt)
			div58 = Rtt(ht, wt, 90, div58)

			div59 = resized[co_l:co_m, co_b:co_c][:]
			div59 = Bluring(div59, bl, ht, wt)
			div59 = Rtt(ht, wt, 180, div59)

			div60 = resized[co_l:co_m, co_p:co_q][:]
			div60 = Bluring(div60, bl, ht, wt)
			div60 = Rtt(ht, wt, -90, div60)

			div61 = resized[co_l:co_m, co_q:co_r][:]
			div61 = Bluring(div61, bl, ht, wt)
			div61 = Rtt(ht, wt, -90, div61)

			div62 = resized[co_m:co_n, co_b:co_c][:]
			div62 = Bluring(div62, bl, ht, wt)
			div62 = Rtt(ht, wt, 90, div62)

			div63 = resized[co_m:co_n, co_c:co_d][:]
			div63 = Bluring(div63, bl, ht, wt)

			div64 = resized[co_m:co_n, co_o:co_p][:]
			div64 = Bluring(div64, bl, ht, wt)
			div64 = Rtt(ht, wt, -90, div64)

			div65 = resized[co_m:co_n, co_p:co_q][:]
			div65 = Bluring(div65, bl, ht, wt)
			div65 = Rtt(ht, wt, -90, div65)

			div66 = resized[co_n:co_o, co_b:co_c][:]
			div66 = Bluring(div66, bl, ht, wt)
			div66 = Rtt(ht, wt, -90, div66)

			div67 = resized[co_n:co_o, co_c:co_d][:]
			div67 = Bluring(div67, bl, ht, wt)
			div67 = Rtt(ht, wt, 90, div67)

			div68 = resized[co_n:co_o, co_d:co_e][:]
			div68 = Bluring(div68, bl, ht, wt)
			div68 = Rtt(ht, wt, 180, div68)

			div69 = resized[co_n:co_o, co_n:co_o][:]
			div69 = Bluring(div69, bl, ht, wt)
			div69 = Rtt(ht, wt, 180, div69)

			div70 = resized[co_n:co_o, co_o:co_p][:]
			div70 = Bluring(div70, bl, ht, wt)
			div70 = Rtt(ht, wt, -90, div70)

			div71 = resized[co_n:co_o, co_p:co_q][:]
			div71 = Bluring(div71, bl, ht, wt)
			div71 = Rtt(ht, wt, -90, div71)

			div72 = resized[co_o:co_p, co_c:co_d][:]
			div72 = Bluring(div72, bl, ht, wt)
			div72 = Rtt(ht, wt, 90, div72)

			div73 = resized[co_o:co_p, co_d:co_e][:]
			div73 = Bluring(div73, bl, ht, wt)
			div73 = Rtt(ht, wt, 180, div73)

			div74 = resized[co_o:co_p, co_e:co_f][:]
			div74 = Bluring(div74, bl, ht, wt)
			div74 = Rtt(ht, wt, -90, div74)

			div75 = resized[co_o:co_p, co_m:co_n][:]
			div75 = Bluring(div75, bl, ht, wt)
			div75 = Rtt(ht, wt, 180, div75)

			div76 = resized[co_o:co_p, co_n:co_o][:]
			div76 = Bluring(div76, bl, ht, wt)
			div76 = Rtt(ht, wt, 90, div76)

			div77 = resized[co_o:co_p, co_o:co_p][:]
			div77 = Bluring(div77, bl, ht, wt)
			div77 = Rtt(ht, wt, 90, div77)

			div78 = resized[co_p:co_q, co_d:co_e][:]
			div78 = Bluring(div78, bl, ht, wt)

			div79 = resized[co_p:co_q, co_e:co_f][:]
			div79 = Bluring(div79, bl, ht, wt)
			div79 = Rtt(ht, wt, -90, div79)

			div80 = resized[co_p:co_q, co_f:co_g][:]
			div80 = Bluring(div80, bl, ht, wt)
			div80 = Rtt(ht, wt, 180, div80)

			div81 = resized[co_p:co_q, co_g:co_h][:]
			div81 = Bluring(div81, bl, ht, wt)
			div81 = Rtt(ht, wt, 180, div81)

			div82 = resized[co_p:co_q, co_h:co_i][:]
			div82 = Bluring(div82, bl, ht, wt)

			div83 = resized[co_p:co_q, co_i:co_j][:]
			div83 = Bluring(div83, bl, ht, wt)
			div83 = Rtt(ht, wt, 90, div83)

			div84 = resized[co_p:co_q, co_j:co_k][:]
			div84 = Bluring(div84, bl, ht, wt)
			div84 = Rtt(ht, wt, -90, div84)

			div85 = resized[co_p:co_q, co_k:co_l][:]
			div85 = Bluring(div85, bl, ht, wt)

			div86 = resized[co_p:co_q, co_l:co_m][:]
			div86 = Bluring(div86, bl, ht, wt)
			div86 = Rtt(ht, wt, 90, div86)

			div87 = resized[co_p:co_q, co_m:co_n][:]
			div87 = Bluring(div87, bl, ht, wt)
			div87 = Rtt(ht, wt, 180, div87)

			div88 = resized[co_p:co_q, co_n:co_o][:]
			div88 = Bluring(div88, bl, ht, wt)
			div88 = Rtt(ht, wt, 180, div88)

			div89 = resized[co_q:co_r, co_f:co_g][:]
			div89 = Bluring(div89, bl, ht, wt)
			div89 = Rtt(ht, wt, -90, div89)

			div90 = resized[co_q:co_r, co_g:co_h][:]
			div90 = Bluring(div90, bl, ht, wt)
			div90 = Rtt(ht, wt, 90, div90)

			div91 = resized[co_q:co_r, co_h:co_i][:]
			div91 = Bluring(div91, bl, ht, wt)
			div91 = Rtt(ht, wt, -90, div91)

			div92 = resized[co_q:co_r, co_i:co_j][:]
			div92 = Bluring(div92, bl, ht, wt)

			div93 = resized[co_q:co_r, co_j:co_k][:]
			div93 = Bluring(div93, bl, ht, wt)
			div93 = Rtt(ht, wt, 180, div93)

			div94 = resized[co_q:co_r, co_k:co_l][:]
			div94 = Bluring(div94, bl, ht, wt)
			div94 = Rtt(ht, wt, 90, div94)

			div95 = resized[co_q:co_r, co_l:co_m][:]
			div95 = Bluring(div95, bl, ht, wt)
			div95 = Rtt(ht, wt, -90, div95)

			resized[co_b:co_c, co_d:co_e] = div01
			Cnct(resized, co_b - 2, co_b + 2, co_d, co_e)
			Cnct(resized, co_b, co_c, co_d - 2, co_d + 2)
			resized[co_b:co_c, co_e:co_f] = div10
			Cnct(resized, co_b - 2, co_b + 21, co_e, co_f)
			Cnct(resized, co_b, co_c, co_e - 2, co_e + 2)
			resized[co_b:co_c, co_f:co_g] = div04
			Cnct(resized, co_b - 2, co_b + 2, co_f, co_g)
			Cnct(resized, co_b, co_c, co_f - 2, co_f + 2)
			Cnct(resized, co_c - 2, co_c + 2, co_f, co_g)
			resized[co_b:co_c, co_g:co_h] = div07
			Cnct(resized, co_b - 2, co_b + 2, co_g, co_h)
			Cnct(resized, co_b, co_c, co_g - 2, co_g + 2)
			Cnct(resized, co_c - 2, co_c + 2, co_g, co_h)
			resized[co_b:co_c, co_h:co_i] = div23
			Cnct(resized, co_b - 2, co_b + 2, co_h, co_i)
			Cnct(resized, co_b, co_c, co_h - 2, co_h + 2)
			Cnct(resized, co_c - 2, co_c + 2, co_h, co_i)
			resized[co_b:co_c, co_i:co_j] = div16
			Cnct(resized, co_b - 2, co_b + 2, co_i, co_j)
			Cnct(resized, co_b, co_c, co_i - 2, co_i + 2)
			Cnct(resized, co_c - 2, co_c + 2, co_i, co_j)
			resized[co_b:co_c, co_j:co_k] = div18
			Cnct(resized, co_b - 2, co_b + 2, co_j, co_k)
			Cnct(resized, co_b, co_c, co_j - 2, co_j + 2)
			Cnct(resized, co_c - 2, co_c + 2, co_j, co_k)
			resized[co_b:co_c, co_k:co_l] = div08
			Cnct(resized, co_b - 2, co_b + 2, co_k, co_l)
			Cnct(resized, co_b, co_c, co_k - 2, co_k + 2)
			Cnct(resized, co_c - 2, co_c + 2, co_k, co_l)
			resized[co_b:co_c, co_l:co_m] = div13
			Cnct(resized, co_b - 2, co_b + 2, co_l, co_m)
			Cnct(resized, co_b, co_c, co_l - 2, co_l + 2)
			Cnct(resized, co_c - 2, co_c + 2, co_l, co_m)
			resized[co_b:co_c, co_m:co_n] = div15
			Cnct(resized, co_b - 2, co_b + 2, co_m, co_n)
			Cnct(resized, co_b, co_c, co_m - 2, co_m + 2)
			resized[co_b:co_c, co_n:co_o] = div06
			Cnct(resized, co_b - 2, co_b + 2, co_n, co_o)
			Cnct(resized, co_b, co_c, co_n - 2, co_n + 2)
			Cnct(resized, co_b, co_c, co_o - 2, co_o + 2)

			resized[co_c:co_d, co_c:co_d] = div09
			Cnct(resized, co_c - 2, co_c + 2, co_c, co_d)
			Cnct(resized, co_c, co_d, co_c - 2, co_c + 2)
			resized[co_c:co_d, co_d:co_e] = div20
			Cnct(resized, co_c - 2, co_c + 2, co_d, co_e)
			Cnct(resized, co_c, co_d, co_d - 2, co_d + 2)
			resized[co_c:co_d, co_e:co_f] = div19
			Cnct(resized, co_c - 2, co_c + 2, co_e, co_f)
			Cnct(resized, co_c, co_d, co_e - 2, co_e + 2)
			Cnct(resized, co_c, co_d, co_f - 2, co_f + 2)
			Cnct(resized, co_d - 2, co_d + 2, co_e, co_f)

			resized[co_c:co_d, co_m:co_n] = div22
			Cnct(resized, co_c - 2, co_c + 2, co_m, co_n)
			Cnct(resized, co_c, co_d, co_m - 2, co_m + 2)
			Cnct(resized, co_d - 2, co_d + 2, co_m, co_n)
			resized[co_c:co_d, co_n:co_o] = div12
			Cnct(resized, co_c - 2, co_c + 2, co_n, co_o)
			Cnct(resized, co_c, co_d, co_n - 2, co_n + 2)
			resized[co_c:co_d, co_o:co_p] = div26
			Cnct(resized, co_c - 2, co_c + 2, co_o, co_p)
			Cnct(resized, co_c, co_d, co_o - 2, co_o + 2)
			Cnct(resized, co_c, co_d, co_p - 2, co_p + 2)

			resized[co_d:co_e, co_b:co_c] = div25
			Cnct(resized, co_d - 2, co_d + 2, co_b, co_c)
			Cnct(resized, co_d, co_e, co_b - 2, co_b + 2)
			resized[co_d:co_e, co_c:co_d] = div24
			Cnct(resized, co_d - 2, co_d + 2, co_c, co_d)
			Cnct(resized, co_d, co_e, co_c - 2, co_c + 2)
			resized[co_d:co_e, co_d:co_e] = div21
			Cnct(resized, co_d - 2, co_d + 2, co_d, co_e)
			Cnct(resized, co_d, co_e, co_d - 2, co_d + 2)
			Cnct(resized, co_d, co_e, co_e - 2, co_e + 2)
			Cnct(resized, co_e - 2, co_e + 2, co_d, co_e)

			resized[co_d:co_e, co_n:co_o] = div28
			Cnct(resized, co_d - 2, co_d + 2, co_n, co_o)
			Cnct(resized, co_d, co_e, co_n - 2, co_n + 2)
			Cnct(resized, co_e - 2, co_e + 2, co_n, co_o)
			resized[co_d:co_e, co_o:co_p] = div91
			Cnct(resized, co_d - 2, co_d + 2, co_o, co_p)
			Cnct(resized, co_d, co_e, co_o - 2, co_o + 2)
			resized[co_d:co_e, co_p:co_q] = div31
			Cnct(resized, co_d - 2, co_d + 2, co_p, co_q)
			Cnct(resized, co_d, co_e, co_p - 2, co_p + 2)
			Cnct(resized, co_d, co_e, co_q - 2, co_q + 2)

			resized[co_e:co_f, co_b:co_c] = div27
			Cnct(resized, co_e - 2, co_e + 2, co_b, co_c)
			Cnct(resized, co_e, co_f, co_b - 2, co_b + 2)
			resized[co_e:co_f, co_c:co_d] = div30
			Cnct(resized, co_e - 2, co_e + 2, co_c, co_d)
			Cnct(resized, co_e, co_f, co_c - 2, co_c + 2)
			Cnct(resized, co_e, co_f, co_d - 2, co_d + 2)
			Cnct(resized, co_f - 2, co_f + 2, co_c, co_d)

			resized[co_e:co_f, co_o:co_p] = div37
			Cnct(resized, co_e - 2, co_e + 2, co_o, co_p)
			Cnct(resized, co_e, co_f, co_o - 2, co_o + 2)
			Cnct(resized, co_f - 2, co_f + 2, co_o, co_p)
			resized[co_e:co_f, co_p:co_q] = div34
			Cnct(resized, co_e - 2, co_e + 2, co_p, co_q)
			Cnct(resized, co_e, co_f, co_p - 2, co_p + 2)
			Cnct(resized, co_e, co_f, co_q - 2, co_q + 2)

			resized[co_f:co_g, co_a:co_b] = div32
			Cnct(resized, co_f - 2, co_f + 2, co_a, co_b)
			Cnct(resized, co_f, co_g, co_a - 2, co_a + 2)
			resized[co_f:co_g, co_b:co_c] = div35
			Cnct(resized, co_f - 2, co_f + 2, co_b, co_c)
			Cnct(resized, co_f, co_g, co_b - 2, co_b + 2)
			Cnct(resized, co_f, co_g, co_c - 2, co_c + 2)

			resized[co_f:co_g, co_p:co_q] = div93
			Cnct(resized, co_f - 2, co_f + 2, co_p, co_q)
			Cnct(resized, co_f, co_g, co_p - 2, co_p + 2)
			resized[co_f:co_g, co_q:co_r] = div43
			Cnct(resized, co_f - 2, co_f + 2, co_q, co_r)
			Cnct(resized, co_f, co_g, co_q - 2, co_q + 2)
			Cnct(resized, co_f, co_g, co_r - 2, co_r + 2)

			resized[co_g:co_h, co_a:co_b] = div36
			Cnct(resized, co_g - 2, co_g + 2, co_a, co_b)
			Cnct(resized, co_g, co_h, co_a - 2, co_a + 2)
			resized[co_g:co_h, co_b:co_c] = div39
			Cnct(resized, co_g - 2, co_g + 2, co_b, co_c)
			Cnct(resized, co_g, co_h, co_b - 2, co_b + 2)
			Cnct(resized, co_g, co_h, co_c - 2, co_c + 2)

			resized[co_g:co_h, co_p:co_q] = div40
			Cnct(resized, co_g - 2, co_g + 2, co_p, co_q)
			Cnct(resized, co_g, co_h, co_p - 2, co_p + 2)
			resized[co_g:co_h, co_q:co_r] = div42
			Cnct(resized, co_g - 2, co_g + 2, co_q, co_r)
			Cnct(resized, co_g, co_h, co_q - 2, co_q + 2)
			Cnct(resized, co_g, co_h, co_r - 2, co_r + 2)

			resized[co_h:co_i, co_a:co_b] = div91
			Cnct(resized, co_h - 2, co_h + 2, co_a, co_b)
			Cnct(resized, co_h, co_i, co_a - 2, co_a + 2)
			resized[co_h:co_i, co_b:co_c] = div38
			Cnct(resized, co_h - 2, co_h + 2, co_b, co_c)
			Cnct(resized, co_h, co_i, co_b - 2, co_b + 2)
			Cnct(resized, co_h, co_i, co_c - 2, co_c + 2)

			resized[co_h:co_i, co_p:co_q] = div85
			Cnct(resized, co_h - 2, co_h + 2, co_p, co_q)
			Cnct(resized, co_h, co_i, co_p - 2, co_p + 2)
			resized[co_h:co_i, co_q:co_r] = div47
			Cnct(resized, co_h - 2, co_h + 2, co_q, co_r)
			Cnct(resized, co_h, co_i, co_q - 2, co_q + 2)
			Cnct(resized, co_h, co_i, co_r - 2, co_r + 2)

			resized[co_i:co_j, co_a:co_b] = div44
			Cnct(resized, co_i - 2, co_i + 2, co_a, co_b)
			Cnct(resized, co_i, co_j, co_a - 2, co_a + 2)
			resized[co_i:co_j, co_b:co_c] = div12
			Cnct(resized, co_i - 2, co_i + 2, co_b, co_c)
			Cnct(resized, co_i, co_j, co_b - 2, co_b + 2)
			Cnct(resized, co_i, co_j, co_c - 2, co_c + 2)

			resized[co_i:co_j, co_p:co_q] = div48
			Cnct(resized, co_i - 2, co_i + 2, co_p, co_q)
			Cnct(resized, co_i, co_j, co_p - 2, co_p + 2)
			resized[co_i:co_j, co_q:co_r] = div87
			Cnct(resized, co_i - 2, co_i + 2, co_q, co_r)
			Cnct(resized, co_i, co_j, co_q - 2, co_q + 2)
			Cnct(resized, co_i, co_j, co_r - 2, co_r + 2)

			resized[co_j:co_k, co_a:co_b] = div79
			Cnct(resized, co_j - 2, co_j + 2, co_a, co_b)
			Cnct(resized, co_j, co_k, co_a - 2, co_a + 2)
			resized[co_j:co_k, co_b:co_c] = div17
			Cnct(resized, co_j - 2, co_j + 2, co_b, co_c)
			Cnct(resized, co_j, co_k, co_b - 2, co_b + 2)
			Cnct(resized, co_j, co_k, co_c - 2, co_c + 2)

			resized[co_j:co_k, co_p:co_q] = div85
			Cnct(resized, co_j - 2, co_j + 2, co_p, co_q)
			Cnct(resized, co_j, co_k, co_p - 2, co_p + 2)
			resized[co_j:co_k, co_q:co_r] = div93
			Cnct(resized, co_j - 2, co_j + 2, co_q, co_r)
			Cnct(resized, co_j, co_k, co_q - 2, co_q + 2)
			Cnct(resized, co_j, co_k, co_r - 2, co_r + 2)

			resized[co_k:co_l, co_a:co_b] = div91
			Cnct(resized, co_k - 2, co_k + 2, co_a, co_b)
			Cnct(resized, co_k, co_l, co_a - 2, co_a + 2)
			resized[co_k:co_l, co_b:co_c] = div33
			Cnct(resized, co_k - 2, co_k + 2, co_b, co_c)
			Cnct(resized, co_k, co_l, co_b - 2, co_b + 2)
			Cnct(resized, co_k, co_l, co_c - 2, co_c + 2)

			resized[co_k:co_l, co_p:co_q] = div93
			Cnct(resized, co_k - 2, co_k + 2, co_p, co_q)
			Cnct(resized, co_k, co_l, co_p - 2, co_p + 2)
			resized[co_k:co_l, co_q:co_r] = div71
			Cnct(resized, co_k - 2, co_k + 2, co_q, co_r)
			Cnct(resized, co_k, co_l, co_q - 2, co_q + 2)
			Cnct(resized, co_k, co_l, co_r - 2, co_r + 2)

			resized[co_l:co_m, co_a:co_b] = div85
			Cnct(resized, co_l - 2, co_l + 2, co_a, co_b)
			Cnct(resized, co_l, co_m, co_a - 2, co_a + 2)
			Cnct(resized, co_m - 2, co_m + 2, co_a, co_b)
			resized[co_l:co_m, co_b:co_c] = div55
			Cnct(resized, co_l - 2, co_l + 2, co_b, co_c)
			Cnct(resized, co_l, co_m, co_b - 2, co_b + 2)
			Cnct(resized, co_l, co_m, co_c - 2, co_c + 2)

			resized[co_l:co_m, co_p:co_q] = div19
			Cnct(resized, co_l - 2, co_l + 2, co_p, co_q)
			Cnct(resized, co_l, co_m, co_p - 2, co_p + 2)
			resized[co_l:co_m, co_q:co_r] = div87
			Cnct(resized, co_l - 2, co_l + 2, co_q, co_r)
			Cnct(resized, co_l, co_m, co_q - 2, co_q + 2)
			Cnct(resized, co_l, co_m, co_r - 2, co_r + 2)
			Cnct(resized, co_m - 2, co_m + 2, co_q, co_r)

			resized[co_m:co_n, co_b:co_c] = div87
			Cnct(resized, co_m - 2, co_m + 2, co_b, co_c)
			Cnct(resized, co_m, co_n, co_b - 2, co_b + 2)
			resized[co_m:co_n, co_c:co_d] = div85
			Cnct(resized, co_m - 2, co_m + 2, co_c, co_d)
			Cnct(resized, co_m, co_n, co_c - 2, co_c + 2)
			Cnct(resized, co_m, co_n, co_d - 2, co_d + 2)

			resized[co_m:co_n, co_o:co_p] = div91
			Cnct(resized, co_m - 2, co_m + 2, co_o, co_p)
			Cnct(resized, co_m, co_n, co_o - 2, co_o + 2)
			resized[co_m:co_n, co_p:co_q] = div93
			Cnct(resized, co_m - 2, co_m + 2, co_p, co_q)
			Cnct(resized, co_m, co_n, co_p - 2, co_p + 2)
			Cnct(resized, co_m, co_n, co_q - 2, co_q + 2)

			resized[co_n:co_o, co_b:co_c] = div68
			Cnct(resized, co_n - 2, co_n + 2, co_b, co_c)
			Cnct(resized, co_n, co_o, co_b - 2, co_b + 2)
			Cnct(resized, co_o - 2, co_o + 2, co_b, co_c)
			resized[co_n:co_o, co_c:co_d] = div74
			Cnct(resized, co_n - 2, co_n + 2, co_c, co_d)
			Cnct(resized, co_n, co_o, co_c - 2, co_c + 2)
			resized[co_n:co_o, co_d:co_e] = div87
			Cnct(resized, co_n - 2, co_n + 2, co_d, co_e)
			Cnct(resized, co_n, co_o, co_d - 2, co_d + 2)
			Cnct(resized, co_n, co_o, co_e - 2, co_e + 2)

			resized[co_n:co_o, co_n:co_o] = div87
			Cnct(resized, co_n - 2, co_n + 2, co_n, co_o)
			Cnct(resized, co_n, co_o, co_n - 2, co_n + 2)
			resized[co_n:co_o, co_o:co_p] = div65
			Cnct(resized, co_n - 2, co_n + 2, co_o, co_p)
			Cnct(resized, co_n, co_o, co_o - 2, co_o + 2)
			resized[co_n:co_o, co_p:co_q] = div45
			Cnct(resized, co_n - 2, co_n + 2, co_p, co_q)
			Cnct(resized, co_n, co_o, co_p - 2, co_p + 2)
			Cnct(resized, co_n, co_o, co_q - 2, co_q + 2)
			Cnct(resized, co_o - 2, co_o + 2, co_p, co_q)

			resized[co_o:co_p, co_c:co_d] = div72
			Cnct(resized, co_o - 2, co_o + 2, co_c, co_d)
			Cnct(resized, co_o, co_p, co_c - 2, co_c + 2)
			Cnct(resized, co_p - 2, co_p + 2, co_c, co_d)
			resized[co_o:co_p, co_d:co_e] = div89
			Cnct(resized, co_o - 2, co_o + 2, co_d, co_e)
			Cnct(resized, co_o, co_p, co_d - 2, co_d + 2)
			resized[co_o:co_p, co_e:co_f] = div73
			Cnct(resized, co_o - 2, co_o + 2, co_e, co_f)
			Cnct(resized, co_o, co_p, co_e - 2, co_e + 2)
			Cnct(resized, co_o, co_p, co_f - 2, co_f + 2)

			resized[co_o:co_p, co_m:co_n] = div78
			Cnct(resized, co_o - 2, co_o + 2, co_m, co_n)
			Cnct(resized, co_o, co_p, co_m - 2, co_m + 2)
			resized[co_o:co_p, co_n:co_o] = div51
			Cnct(resized, co_o - 2, co_o + 2, co_n, co_o)
			Cnct(resized, co_o, co_p, co_n - 2, co_n + 2)
			resized[co_o:co_p, co_o:co_p] = div76
			Cnct(resized, co_o - 2, co_o + 2, co_o, co_p)
			Cnct(resized, co_o, co_p, co_o - 2, co_o + 2)
			Cnct(resized, co_o, co_p, co_p - 2, co_p + 2)
			Cnct(resized, co_p - 2, co_p + 2, co_o, co_p)

			resized[co_p:co_q, co_d:co_e] = div75
			Cnct(resized, co_p - 2, co_p + 2, co_d, co_e)
			Cnct(resized, co_p, co_q, co_d - 2, co_d + 2)
			Cnct(resized, co_q - 2, co_q + 2, co_d, co_e)
			resized[co_p:co_q, co_e:co_f] = div52
			Cnct(resized, co_p - 2, co_p + 2, co_e, co_f)
			Cnct(resized, co_p, co_q, co_e - 2, co_e + 2)
			Cnct(resized, co_q - 2, co_q + 2, co_e, co_f)
			resized[co_p:co_q, co_f:co_g] = div81
			Cnct(resized, co_p - 2, co_p + 2, co_f, co_g)
			Cnct(resized, co_p, co_q, co_f - 2, co_f + 2)
			resized[co_p:co_q, co_g:co_h] = div55
			Cnct(resized, co_p - 2, co_p + 2, co_g, co_h)
			Cnct(resized, co_p, co_q, co_g - 2, co_g + 2)
			resized[co_p:co_q, co_h:co_i] = div95
			Cnct(resized, co_p - 2, co_p + 2, co_h, co_i)
			Cnct(resized, co_p, co_q, co_h - 2, co_h + 2)
			resized[co_p:co_q, co_i:co_j] = div51
			Cnct(resized, co_p - 2, co_p + 2, co_i, co_j)
			Cnct(resized, co_p, co_q, co_i - 2, co_i + 2)
			resized[co_p:co_q, co_j:co_k] = div54
			Cnct(resized, co_p - 2, co_p + 2, co_j, co_k)
			Cnct(resized, co_p, co_q, co_j - 2, co_j + 2)
			resized[co_p:co_q, co_k:co_l] = div80
			Cnct(resized, co_p - 2, co_p + 2, co_k, co_l)
			Cnct(resized, co_p, co_q, co_k - 2, co_k + 2)
			resized[co_p:co_q, co_l:co_m] = div79
			Cnct(resized, co_p - 2, co_p + 2, co_l, co_m)
			Cnct(resized, co_p, co_q, co_l - 2, co_l + 2)
			resized[co_p:co_q, co_m:co_n] = div52
			Cnct(resized, co_p - 2, co_p + 2, co_m, co_n)
			Cnct(resized, co_p, co_q, co_m - 2, co_m + 2)
			Cnct(resized, co_q - 2, co_q + 2, co_m, co_n)
			resized[co_p:co_q, co_n:co_o] = div94
			Cnct(resized, co_p - 2, co_p + 2, co_n, co_o)
			Cnct(resized, co_p, co_q, co_n - 2, co_n + 2)
			Cnct(resized, co_p, co_q, co_o - 2, co_o + 2)
			Cnct(resized, co_q - 2, co_q + 2, co_n, co_o)

			resized[co_q:co_r, co_f:co_g] = div51
			Cnct(resized, co_q - 2, co_q + 2, co_f, co_g)
			Cnct(resized, co_q, co_r, co_f - 2, co_f + 2)
			Cnct(resized, co_r - 2, co_r + 2, co_f, co_g)
			resized[co_q:co_r, co_g:co_h] = div91
			Cnct(resized, co_q - 2, co_q + 2, co_g, co_h)
			Cnct(resized, co_q, co_r, co_g - 2, co_g + 2)
			Cnct(resized, co_r - 2, co_r + 2, co_g, co_h)
			resized[co_q:co_r, co_h:co_i] = div55
			Cnct(resized, co_q - 2, co_q + 2, co_h, co_i)
			Cnct(resized, co_q, co_r, co_h - 2, co_h + 2)
			Cnct(resized, co_r - 2, co_r + 2, co_h, co_i)
			resized[co_q:co_r, co_i:co_j] = div93
			Cnct(resized, co_q - 2, co_q + 2, co_i, co_j)
			Cnct(resized, co_q, co_r, co_i - 2, co_i + 2)
			Cnct(resized, co_r - 2, co_r + 2, co_i, co_j)
			resized[co_q:co_r, co_j:co_k] = div67
			Cnct(resized, co_q - 2, co_q + 2, co_j, co_k)
			Cnct(resized, co_q, co_r, co_j - 2, co_j + 2)
			Cnct(resized, co_r - 2, co_r + 2, co_j, co_k)
			resized[co_q:co_r, co_k:co_l] = div83
			Cnct(resized, co_q - 2, co_q + 2, co_k, co_l)
			Cnct(resized, co_q, co_r, co_k - 2, co_k + 2)
			Cnct(resized, co_r - 2, co_r + 2, co_k, co_l)
			resized[co_q:co_r, co_l:co_m] = div53
			Cnct(resized, co_q - 2, co_q + 2, co_l, co_m)
			Cnct(resized, co_q, co_r, co_l - 2, co_l + 2)
			Cnct(resized, co_q, co_r, co_m - 2, co_m + 2)
			Cnct(resized, co_r - 2, co_r + 2, co_l, co_m)

			if pred_ht > 200:
				changed = cv2.resize(resized, (pred_wt, pred_ht), interpolation=cv2.INTER_CUBIC)
			else:
				changed = cv2.resize(resized, (pred_wt, pred_ht), interpolation=cv2.INTER_AREA)

			imgcv[y_offset1:y_offset2, x_offset1:x_offset2] = changed

		cv2.imwrite(img_name, imgcv)

		# 복사를 위한
		copy_src_path = img_name
		copy_dst_path = self.FLAGS.imgdir  # os.path.join(self.FLAGS.imgdir, os.path.basename(im))

		# 원본 쪽에 복사
		shutil.copy(copy_src_path, copy_dst_path)
	'''
	# 좌표 복사(이미지)
	dst_img = []
	for i in result_coord:
		# left, top, right, bot
		temp_x = (i[2] - i[0]) // 3;
		temp_y = (i[3] - i[1]) // 3;
		print('temp-x: ', temp_x)
		print('temp-y: ', temp_y)

		dst = imgcv.copy()
		# top-bot, left-right
		# dst_img.append(dst[i[1]:i[3], i[0]:i[2]])


		dst = cv2.GaussianBlur(dst, (5,5), 0)


		# dst = cv2.GaussianBlur(dst, (5, 5), 0)
		dst_img.append(dst[i[1]+temp_y:i[3]-temp_y, i[0]+temp_x:i[2]-temp_x])

		# 박스 크기 보정

	# 디텍션 이미지의 정보 / 빈 이미지 생성
	height, width, channel = imgcv.shape
	print('복사 이미지 X 크기: {0}, Y 크기: {1}'.format(width, height))
	tempImg = np.zeros((height, width, 3), np.uint8)

	# 영역 부분만을 빈 이미지에 합침
	num = 0
	idst = imgcv.copy()
	for i in result_coord:
		# left, top, right, bot
		temp_x = (i[2] - i[0]) // 3;
		temp_y = (i[3] - i[1]) // 3;

		x_offset = i[0] + temp_x
		y_offset = i[1] + temp_y

		idst[y_offset:y_offset + dst_img[num].shape[0], x_offset:x_offset + dst_img[num].shape[1]] = dst_img[num]
		tempImg[y_offset:y_offset + dst_img[num].shape[0], x_offset:x_offset + dst_img[num].shape[1]] = dst_img[num]
		num = num + 1
	'''

	# 디텍션 결과 저장
	# cv2.imwrite(img_name, imgcv)

	'''
	# 디텍션 이후 좌표 부분만 저장
	cv2.imwrite(os.path.splitext(img_name)[0] + '_temp2.jpg', idst)
	cv2.imwrite(os.path.splitext(img_name)[0] + '_temp.jpg', tempImg)
	'''
