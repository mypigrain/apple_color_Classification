import tkinter
import cv2
import numpy as np
from tkinter import filedialog
from tkinter import ttk

#显示长和宽
width = 800
height = 600

#预处理按钮长宽
pre_relwidth = 1/7
pre_relheight = 0.2

#形态学操作按钮长宽
mor_relwidth = 1/4
mor_relheight = 0.3

#最终处理按钮长宽
fin_relwidth = 1/6
fin_relheight = 0.2

global img_local,img_get,img_camera,path,video
video = 0


#########################################图像操作区######################################################
#滤波    
def filt(img):
    blurtype = blur_type.get()
    blurweight = blur_weight.get()
    blurweight = int(blurweight)
    boxsize = blur_box.get()
    boxsize = int(boxsize)
    if blurtype == 1:
        img_blur = cv2.blur(img, (boxsize,boxsize))
    else:
        if blurtype == 2:
            if boxsize % 2 == 1:
                img_blur = cv2.medianBlur(img, boxsize)
            else:
                img_blur = cv2.medianBlur(img, boxsize+1)
        else:
            if boxsize % 2 == 1:   
                img_blur = cv2.GaussianBlur(img,(boxsize,boxsize),blurweight)
            else:
                img_blur = cv2.GaussianBlur(img,(boxsize+1,boxsize+1),blurweight)
    
    return img_blur

#对比度增强
def equal_color(img):
    th = clahe_th.get()
    th = float(th)
    size = clahe_box.get()
    size = int(size)
    clahe = cv2.createCLAHE(clipLimit = th,tileGridSize = (size,size))
    b,g,r = cv2.split(img)
    gh = clahe.apply(g)
    rh = clahe.apply(r)
    img_equal_color= cv2.merge((b,gh,rh))
    return img_equal_color

    
#灰度图
def img_gray(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    return img_gray
    
    
#rgb分量图
def rgb(img):
    imgcur = img.copy()
    b,g,r = cv2.split(imgcur)
    return b,g,r
    
#H分量
def hsl(img):
    enlarge = h_enlarge.get()
    enlarge = int(enlarge)
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,l = cv2.split(hsl)
    h = h*enlarge
    return h
    
    
#二值化
def counter(img):
    thre = threshold.get()
    thre = int(thre)
    ret, th1 = cv2.threshold(img, thre, 255, cv2.THRESH_BINARY_INV)
    return th1

#腐蚀
def erode(img):
    boxsize = mor_box.get()
    times = mor_times.get()
    boxsize = int(boxsize)
    times = int(times)
    kernel = np.ones((boxsize,boxsize),np.uint8)
    dst = cv2.erode(img,kernel,iterations = times)
    return dst

#膨胀
def dilate(img):
    boxsize = mor_box.get()
    times = mor_times.get()
    boxsize = int(boxsize)
    times = int(times)
    kernel = np.ones((boxsize,boxsize),np.uint8)
    dst = cv2.dilate(img,kernel,iterations = times)
    return dst

#开运算
def _open(img):
    boxsize = mor_box.get()
    times = mor_times.get()
    boxsize = int(boxsize)
    times = int(times)
    kernel = np.ones((boxsize,boxsize),np.uint8)
    dst = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel,iterations = times)
    return dst

#闭运算
def close(img):
    boxsize = mor_box.get()
    times = mor_times.get()
    boxsize = int(boxsize)
    times = int(times)
    kernel = np.ones((boxsize,boxsize),np.uint8)
    dst = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel,iterations = times)
    return dst
    
    

#在原图上为苹果画出矩形框并加上文字信息
def draw_rec(th1,th2,img):
    binary,contours, h = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    ret = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    sum = np.sum(th1 == 0)
    red = np.sum(th2 == 0)
    colorpercent = '0%'
    level = 'level 4   color percent:'
    if sum == 0:
        bi = 0
        level = 'level 4   color percent:'        
    else:     
        bi = red / sum 
        percent = int(bi*100)
        colorpercent = str(percent)+'%'
        if bi > 0.9:
            level = 'level 1   color percent:'
        else:
            if bi > 0.75:
                level = 'level 2   color percent:'
            else:
                if bi > 0.5:
                    level = 'level 3   color percent:'
                else:
                    level = 'level 4   color percent:'
    text = level+colorpercent
    fin_label.config(text = text)
    img = cv2.putText(img, text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)    






#显示图像
def look(name,img):
    cv2.namedWindow(name,0)
    cv2.resizeWindow(name,width,height)
    cv2.imshow(name,img)
    
##################################################################################################################### 



#########################################tkinter功能区###############################################

#选择文件
def chosefile():
    global path,img_camera,img_local,img_get
    path = tkinter.filedialog.askopenfilename() 
    img_local = cv2.imread(path)
    if deal_chose.get() == 2:
    	img_get = img_camera
    else:
    	img_get = img_local



#查看原图按钮
def fuc_originimg():
    
    img = img_get
    look('originimg',img)

#查看滤波图片按钮
def fuc_blur():
    img = img_get
    img_blur = filt(img)
    look('blur',img_blur)

#查对比度增强图片
def fuc_equal_color():
    img = img_get
    img_equal_color = equal_color(img)
    look('equal_color',img_equal_color)

#查看灰度图
def fuc_img_gray():
    img = img_get
    imggray = img_gray(img)
    look('gray',imggray)

#查看rgb分量
def fuc_rgb():
    img = img_get
    B,G,R = rgb(img)
    look('r',R)
    look('b',B)
    look('g',R)

#查看H分量
def fuc_hsl():
    img = img_get
    H = hsl(img)
    look('H',H)

#查看二值图
def fuc_counter():
    img = img_get
    b,g,r = rgb(img)
    th = counter(r)
    look('counter',th)

#查看腐蚀后图片
def fuc_erode():
    img = img_get
    b,g,r = rgb(img)
    th = counter(r)
    mor = erode(th)
    look('erode',mor)
    
#查看膨胀后图片
def fuc_dilate():
    img = img_get
    b,g,r = rgb(img)
    th = counter(r)
    mor = dilate(th)
    look('dilate',mor)
    
#查看开运算后图片
def fuc_open():
    img = img_get
    b,g,r = rgb(img)
    th = counter(r)
    mor = _open(th)
    look('open',mor)

#查看闭运算后的图片
def fuc_close():
    img = img_get
    b,g,r = rgb(img)
    th = counter(r)
    mor = close(th)
    look('close',mor)

    

#滤波图调试
def blur_debug():
    img = img_get
    blur_debug = filt(img)
    look('blur_debug',blur_debug)
    


#对比度图调试
def equal_debug():
    img = img_get
    blur_debug = filt(img)
    equal_debug = equal_color(blur_debug)
    look('equal_debug',equal_debug)

#H分量调试

def H_debug():
    img = img_get
    blur_debug = filt(img)
    equal_debug = equal_color(blur_debug)
    h_debug = hsl(equal_debug)
    look('h_debug',h_debug)
    
#轮廓二值图调试
def counter_debug():
    img = img_get
    blur_debug = filt(img)
    b,g,r = cv2.split(blur_debug)
    counter_debug = counter(r)
    look('counter_debug',counter_debug)
    
#形态学操作轮廓调试
def mor_c_debug():
    mor_type = mor_chose.get()
    img_chose = mor_debug_chose.get()
    img = img_get
    blur_debug = filt(img)
    b,g,r = cv2.split(blur_debug)   
    thre = counter(r)
    thr = red_threshold.get()
    thr = int(thr)    
    if mor_type == "腐蚀":
        dst = dilate(thre)
    else:
        if mor_type == "膨胀":
            dst = erode(thre)
        else:
            if mor_type == "开运算":
                dst = close(thre)
            else:
                dst = _open(thre)    
    if img_chose == 1:
        th = dst
    else:
        equal_debug = equal_color(blur_debug)
        h_debug = hsl(equal_debug)
        fin = cv2.add(dst,h_debug)
        ret, th = cv2.threshold(fin, thr, 255, cv2.THRESH_BINARY)
        

    
    look('mor',th)
    
#处理的图片选择
def fuc_choseimg():
	global img_local,img_get,img_camera

	if deal_chose.get() == 2:
		img_get = img_camera
	else:
		img_get = img_local

#打开摄像头获取图片按钮
def fuc_camera():
	global img_camera,video
	cap = cv2.VideoCapture(video)
	while 1:
		ret,img = cap.read()
		look('please use "q" to get a picture',img)
		if cv2.waitKey(30) & 0xff == ord('q'):
			break
	cv2.destroyAllWindows()
	cap.release()
	img_camera = img
		

#摄像头选择
def fuc_camera_chose():
	global video
	if camera_chose.get() == "IP":
		video = camera_ip.get()
	else:
		video = 0


#单角度处理
def one_output():
	thr = red_threshold.get()
	thr = int(thr)
	img = img_get
	img_blur = filt(img)
	img_equal = equal_color(img_blur)
	b,g,r = cv2.split(img_blur)
	img_counter = counter(r)
	img_counter = _open(img_counter)
	img_counter = close(img_counter)
	img_h = hsl(img_equal)
	fin = cv2.add(img_counter,img_h)
	ret, img_percent = cv2.threshold(fin, thr, 255, cv2.THRESH_BINARY)
	img_cur = img.copy()
	draw_rec(img_counter,img_percent,img_cur)
	look('final_output',img_cur)



#多角度处理
def mu_outptu():
	global video
	thr = red_threshold.get()
	thr = int(thr)
	cap = cv2.VideoCapture(video)
	fin_colorpercent = 0
	while 1:
		ret,img = cap.read()
		look('please use "q" to get a picture and use "w" to end',img)
		get = cv2.waitKey(20)
		if get & 0xff == ord('q'):
			img_blur = filt(img)
			img_equal = equal_color(img_blur)
			b,g,r = cv2.split(img_blur)
			img_counter = counter(r)
			img_counter = _open(img_counter)
			img_counter = close(img_counter)
			img_h = hsl(img_equal)
			fin = cv2.add(img_counter,img_h)
			ret, img_percent = cv2.threshold(fin, thr, 255, cv2.THRESH_BINARY)
			sum = np.sum(img_counter == 0)
			red = np.sum(img_percent == 0)
			if sum != 0:
				percent = red/sum
				colorpercent = int((red/sum)*100)
				fin_colorpercent = fin_colorpercent + colorpercent
			else:
				pass
		else:
			if 	get & 0xff == ord('w'):
				binary,contours, h = cv2.findContours(img_counter, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
				cnt = contours[-1]
				x, y, w, h = cv2.boundingRect(cnt)
				img_cur = img.copy()
				ret = cv2.rectangle(img_cur, (x, y), (x+w, y+h), (0, 0, 255), 2)     
				text_colorpercent = str(colorpercent)+'%'
				if percent > 0.9:
					level = 'level 1   color percent:'
				else:
					if percent > 0.75:
						level = 'level 2   color percent:'
					else:
						if percent > 0.5:
							level = 'level 3   color percent:'
						else:
							level = 'level 4   color percent:'
				text = level+text_colorpercent
				fin_label.config(text = text)
				img_cur = cv2.putText(img_cur, text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
				break
	cv2.destroyAllWindows()
	cap.release()
	look('mu_outptu',img_cur)





#动态调试
def preprocess_debug():
    pre_swich = pre_debug_on.get()
    pre_img_type = pre_debug_chose.get()
    mor_swich = mor_debug_on.get()
    mor_img_type = mor_debug_chose.get()
    
    if pre_swich == 1:
        if pre_img_type == 1:
            blur_debug()
        else:
            if pre_img_type == 2:
                equal_debug()
            else:
                if pre_img_type == 3:
                    H_debug()
                else:
                    counter_debug()        
    else:
        pass
    
    
    if mor_swich ==1:
        mor_c_debug()
        
        
    else:
        pass

   
    

    

    
    
#########################################################################################################



root = tkinter.Tk()
root.title('苹果色泽分级系统')
root.minsize(500,500)


#调试选择参数
blur_type = tkinter.IntVar()
pre_debug_on = tkinter.IntVar()
pre_debug_chose = tkinter.IntVar()
mor_debug_on = tkinter.IntVar()
mor_debug_chose = tkinter.IntVar()
deal_chose = tkinter.IntVar()


	
#最后输出框架
final = tkinter.LabelFrame(text = '最终处理',fg = 'DarkBlue')
final.place(relx = 0,rely = 0,relwidth = 1,relheight = 0.3)


#预处理框架
preprocess = tkinter.LabelFrame(text = '预处理',fg = 'DarkBlue')    
preprocess.place(relx = 0,rely = 0.3,relwidth = 1,relheight = 0.42)

#形态学操作框架
morphological = tkinter.LabelFrame(text = '形态学操作',fg = 'DarkBlue')
morphological.place(relx = 0,rely = 0.72,relwidth = 1,relheight = 0.28)

#摄像头ip地址
camera_ip = tkinter.Entry(final,width = 35,relief = 'solid')
camera_ip.place(relx = 0.3,rely = 0.77)
tkinter.Label(final,text = 'IP地址: ').place(relx = 0.2,rely = 0.77)
camera_ip.delete(0, "end")
camera_ip.insert("insert",'http://admin:admin@192.168.124.6:8081/' )



#滤波选择
equal = ttk.Radiobutton(preprocess, text = '均值滤波', variable = blur_type, value = 1)
equal.place(relx = 0.01,rely = 0.35)
mid = ttk.Radiobutton(preprocess, text = '中值滤波', variable = blur_type, value = 2)
mid.place(relx = 0.01,rely = 0.55)
gaussian = ttk.Radiobutton(preprocess, text = '高斯滤波', variable = blur_type, value = 3)
gaussian.place(relx = 0.01,rely = 0.75)

#调试图片选择
debug_chose_filt = ttk.Radiobutton(preprocess, text = '滤波图', variable = pre_debug_chose, value = 1)
debug_chose_filt.place(relx = 0.62,rely = 0.55)
debug_chose_equal = ttk.Radiobutton(preprocess, text = '对比度图', variable = pre_debug_chose, value = 2)
debug_chose_equal.place(relx = 0.62,rely = 0.75)
debug_chose_H = ttk.Radiobutton(preprocess, text = 'H分量图', variable = pre_debug_chose, value = 3)
debug_chose_H.place(relx = 0.8,rely = 0.55)
debug_chose_th = ttk.Radiobutton(preprocess, text = '轮廓二值图', variable = pre_debug_chose, value = 4)
debug_chose_th.place(relx = 0.8,rely = 0.75)
debug_chose_counter = ttk.Radiobutton(morphological, text = '轮廓二值图', variable = mor_debug_chose, value = 1)
debug_chose_counter.place(relx = 0.63,rely = 0.64)
debug_chose_thH = ttk.Radiobutton(morphological, text = 'H分量二值图', variable = mor_debug_chose, value = 2)
debug_chose_thH.place(relx = 0.80,rely = 0.64)

#处理图片的选择
deal_local = ttk.Radiobutton(final, text = '处理选择的本地图片', variable = deal_chose, value = 1,command = fuc_choseimg)
deal_local.place(relx = 0.3,rely = 0.55)
deal_camera = ttk.Radiobutton(final, text = '处理从摄像头获取的图片', variable = deal_chose, value = 2,command = fuc_choseimg)
deal_camera.place(relx = 0.62,rely = 0.55)



#滤波盒子大小
blur_box = ttk.Spinbox(preprocess,from_ = 1,to = 15,width = 5,justify = 'center',command = preprocess_debug)
blur_box.place(relx = 0.18,rely = 0.4)
blur_box.delete(0, "end")
blur_box.insert("insert", 5)

#高斯滤波权重大小
blur_weight = ttk.Spinbox(preprocess,from_ = 1,to = 10,width = 5,justify = 'center',command = preprocess_debug)
blur_weight.place(relx = 0.18,rely =0.77)
blur_weight.delete(0, "end")
blur_weight.insert("insert", 2)

#限制对比度的自适应直方图均衡化盒子
clahe_box = ttk.Spinbox(preprocess,from_ = 1,to = 20,width = 5,justify = 'center',command = preprocess_debug)
clahe_box.place(relx = 0.34,rely = 0.4)
clahe_box.delete(0, "end")
clahe_box.insert("insert", 6)

#限制对比度的自适应直方图均衡化阈值
clahe_th = ttk.Spinbox(preprocess,from_ = 1,to = 10,width = 5,justify = 'center',increment = 0.5,command = preprocess_debug)
clahe_th.place(relx = 0.34,rely = 0.77)
clahe_th.delete(0, "end")
clahe_th.insert("insert", 2)

#H放大倍数
h_enlarge = ttk.Spinbox(preprocess,from_ = 1,to = 15,width = 5,justify = 'center',command = preprocess_debug)
h_enlarge.place(relx = 0.50,rely = 0.4)
h_enlarge.delete(0, "end")
h_enlarge.insert("insert", 6)

#二值阈值
threshold = ttk.Spinbox(preprocess,from_ = 0,to = 255,width = 5,justify = 'center',command = preprocess_debug)
threshold.place(relx = 0.50,rely = 0.77)
threshold.delete(0, "end")
threshold.insert("insert",15 )

#形态学操作boxsize
mor_box = ttk.Spinbox(morphological,from_ = 1,to = 15,width = 5,justify = 'center',command = preprocess_debug)
mor_box.place(relx = 0.01,rely = 0.66)
mor_box.delete(0, "end")
mor_box.insert("insert", 5)

#形态学操作次数
mor_times = ttk.Spinbox(morphological,from_ = 0,to = 10,width = 5,justify = 'center',command = preprocess_debug)
mor_times.place(relx = 0.12,rely = 0.66)
mor_times.delete(0, "end")
mor_times.insert("insert",1 )

#H二值阈值
red_threshold = ttk.Spinbox(morphological,from_ = 0,to = 255,width = 5,justify = 'center',command = preprocess_debug)
red_threshold.place(relx = 0.37,rely = 0.66)
red_threshold.delete(0, "end")
red_threshold.insert("insert",115 )

#形态学操作选择
mor_chose = ttk.Spinbox(morphological,width = 7,justify = 'center',wrap=True,values= ("腐蚀", "膨胀", "开运算", "闭运算")\
    ,command = preprocess_debug)
mor_chose.place(relx = 0.23,rely = 0.66)
mor_chose.delete(0, "end")
mor_chose.insert("insert",'开运算' )

#预处理动态调试选择
pre_debug = ttk.Checkbutton(preprocess,variable = pre_debug_on,text = '调试开关')
pre_debug.place(relx = 0.71,rely = 0.35)

#形态学操作动态调试选择
debug = ttk.Checkbutton(morphological,variable = mor_debug_on,text = '调试开关')
debug.place(relx = 0.48,rely = 0.64)

#摄像头选择
camera_chose = ttk.Spinbox(final,width = 7,justify = 'center',wrap=True,values= ("本地", "IP")\
    ,command = fuc_camera_chose)
camera_chose.place(relx = 0.05,rely = 0.77)
camera_chose.delete(0, "end")
camera_chose.insert("insert",'本地' )



#文字设置
pre_label = ttk.Label(preprocess, text='    滤波类型      boxsize         boxsize        \
H放大倍数           动态调试设置区',anchor = "w")
pre_label.place(relx = 0,rely =0.2,relwidth = 1,relheight = 0.15)

blurweight = ttk.Label(preprocess, text='滤波权重')
blurweight.place(relx = 0.18,rely =0.58)

claheth = ttk.Label(preprocess, text='均衡阈值')
claheth.place(relx = 0.34,rely =0.58)

thre_th = ttk.Label(preprocess, text='二值阈值')
thre_th.place(relx = 0.50,rely =0.58)

mor_label = ttk.Label(morphological, text='  boxsize  操作次数  操作类型   H二值阈值\
                   动态调试设置区',anchor = "w")
mor_label.place(relx = 0,rely =0.3,relwidth = 1,relheight = 0.25)

fin_label = tkinter.Label(final, text='分级结果显示区',bg = 'SkyBlue')
fin_label.place(relx = 0,rely =0,relwidth = 1,relheight = 0.25)

fin_chose = ttk.Label(final, text='请选择要处理的图片：')
fin_chose.place(relx = 0.03,rely =0.55)



#原图按钮
originimg_bt = ttk.Button(preprocess,text = '查看原图',command = fuc_originimg)
originimg_bt.place(relx = 0,rely = 0,relwidth = pre_relwidth,relheight = pre_relheight)

#滤波按钮
blur_bt = ttk.Button(preprocess,text = '滤波操作',command = fuc_blur)
blur_bt.place(relx = pre_relwidth,rely = 0,relwidth = pre_relwidth,relheight = pre_relheight)

#对比度增强按钮
equal_bt = ttk.Button(preprocess,text = '对比度增强',command = fuc_equal_color)
equal_bt.place(relx = 2*pre_relwidth,rely = 0,relwidth = pre_relwidth,relheight = pre_relheight)

#灰度图按钮
gray_bt = ttk.Button(preprocess,text = '灰度图',command = fuc_img_gray)
gray_bt.place(relx = 3*pre_relwidth,rely = 0,relwidth = pre_relwidth,relheight = pre_relheight)

#r，g，b按钮
rgb_bt = ttk.Button(preprocess,text = 'rgb分量',command = fuc_rgb)
rgb_bt.place(relx = 4*pre_relwidth,rely = 0,relwidth = pre_relwidth,relheight = pre_relheight)

#H分量按钮
h_bt = ttk.Button(preprocess,text = 'H分量',command = fuc_hsl)
h_bt.place(relx = 5*pre_relwidth,rely = 0,relwidth = pre_relwidth,relheight = pre_relheight)

#二值图按钮
th_bt = ttk.Button(preprocess,text = '二值图',command = fuc_counter)
th_bt.place(relx = 6*pre_relwidth,rely = 0,relwidth = pre_relwidth,relheight = pre_relheight)

#腐蚀按钮
erode_bt = ttk.Button(morphological,text = '腐蚀操作',command = fuc_dilate)
erode_bt.place(relx = 0,rely = 0,relwidth = mor_relwidth,relheight = mor_relheight)

#膨胀按钮
dilate_bt = ttk.Button(morphological,text = '膨胀操作',command = fuc_erode)
dilate_bt.place(relx = mor_relwidth,rely = 0,relwidth = mor_relwidth,relheight = mor_relheight)

#开运算增强按钮
open_bt = ttk.Button(morphological,text = '开运算操作',command = fuc_close)
open_bt.place(relx = 2*mor_relwidth,rely = 0,relwidth = mor_relwidth,relheight = mor_relheight)

#闭运算按钮
close_bt = ttk.Button(morphological,text = '闭运算操作',command = fuc_open)
close_bt.place(relx = 3*mor_relwidth,rely = 0,relwidth = mor_relwidth,relheight = mor_relheight)

#文件选择按钮
chosefile = ttk.Button(final,text = '请选择图片',command = chosefile)
chosefile.place(relx = 0.065,rely = 0.32,relwidth = fin_relwidth,relheight = fin_relheight)

#打开摄像头获取图片按钮
opcam_bt = ttk.Button(final,text = '打开摄像头',command = fuc_camera)
opcam_bt.place(relx = fin_relwidth+0.13,rely = 0.32,relwidth = fin_relwidth,relheight = fin_relheight)

#单角度处理按钮
one_bt = ttk.Button(final,text = '单角度处理',command = one_output)
one_bt.place(relx = 2*fin_relwidth+0.195,rely = 0.32,relwidth = fin_relwidth,relheight = fin_relheight)

#多角度处理按钮
mu_bt = ttk.Button(final,text = '多角度处理',command = mu_outptu)
mu_bt.place(relx = 3*fin_relwidth+0.26,rely = 0.32,relwidth = fin_relwidth,relheight = fin_relheight)

#IP刷新按钮
update_bt = ttk.Button(final,text = '刷新',command = fuc_camera_chose)
update_bt.place(relx = 0.82,rely = 0.76,relwidth = 0.12,relheight = 0.18)




root.mainloop()  
