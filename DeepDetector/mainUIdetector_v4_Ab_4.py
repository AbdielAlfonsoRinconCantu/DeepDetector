import datetime
import time

import imutils
import numpy
from ultralytics import YOLO
import os
import ttkbootstrap as ttk
from ttkbootstrap.constants import  *
from PIL import Image, ImageTk
import threading
import cv2
from collections import defaultdict
from ultralytics.utils.checks import check_imshow, check_requirements
check_requirements("shapely>=2.0.0")
from shapely.geometry import LineString, Point, Polygon
from ultralytics.utils.plotting import Annotator, colors
import queue
import torch
import gc

from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
import numpy as np

print('file: mainUIdetector_v4_Ab_4.py') # mod

#disableDB =  True
devWindows = False

global formato, formatoId, conf, meta, lastTrackId, horaPlot, placas, factor, acumPiezas , metaPieza, promedio, modelPieza, topProm, promTot, promMeta
promedio = 0
lastTrackId = 0

horaPlot = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24]
placas = [0] * 24
factor = [0] * 24
acumPiezas = [0] * 24
metaPieza = [0] * 24
topProm = []
promMeta = []
promTot = []

import sqlite3

connection = sqlite3.connect("deepDb.sqlite3", check_same_thread=False)
cursor = connection.cursor()



################################################################## Video writer



torch.cuda.set_device(0) 

queue = queue.Queue()

#modelPieza.to('cuda')


###############################################################  ventana principal tkinter

root = ttk.Window(themename="cyborg")
root.title("DeepDetector")
root.attributes("-fullscreen", True)

def toogleFullScreenOn(event = None):
    root.attributes("-fullscreen",True)

def toogleFullScreenOff(event = None):
    root.attributes("-fullscreen",False)

root.bind("<Escape>", toogleFullScreenOff)
root.bind("<F11>", toogleFullScreenOn)
################################################################        Menu
frameMenu = ttk.Frame()
frameMenu.pack(side='top')

lbFormato = ttk.Label(frameMenu, text='Formato: ')
lbFormato.pack(side='left')

formatos = cursor.execute("SELECT formato FROM conf").fetchall()
cbFormato = ttk.Combobox(frameMenu, values=formatos,  background='#2148F7')
cbFormato.pack(side= 'left')


lbConf = ttk.Label(frameMenu, text='Confianza: ')
lbConf.pack(side='left')


slConf = ttk.Scale(frameMenu, from_=0, to=100)
slConf.pack(side='left')
slValue = ttk.Label(frameMenu, text = 0)
slValue.pack(side ='left')


def captImg():
    global im88
    print("+++++++++capurando imagen++++++++++")
    nameNow = datetime.datetime.now()
    nameNowStr = nameNow.strftime("%d%H%M%S")
    print(os.path)
    cv2.imwrite(os.path.join(crop_dir_name, nameNowStr + ".png"), im88)

pbCapt = ttk.Button(frameMenu, text = "Capturar", command=captImg)
pbCapt.pack(side = 'left')






def getTurno(now):
    print(now)
    iniTA = now.replace(day=now.day, hour=6, minute=30, second=0, microsecond=0)
    iniTB = now.replace(day=now.day, hour=15, minute=30, second=0, microsecond=0)
    iniTC = now.replace(day=now.day, hour=23, minute=00, second=0, microsecond=0)
    endTC = now.replace(day=now.day - 1, hour=6, minute=30, second=0, microsecond=0)

    if now >= iniTA and now < iniTB:
        turno = 1
    elif now >= iniTB and now < iniTC:
        turno = 2
    elif now >= iniTC or now < iniTA:
        turno = 3
    else:
        turno = 99
    print(turno)
    return turno

def getDBdata():
    global formato,    formatoId, conf, meta, modelPieza

    formato = (cursor.execute("SELECT formato FROM run").fetchone())[0]
    wFormato = "weights/" +formato + ".pt"
    print(formato)
    print(wFormato)
    modelPieza = YOLO(wFormato)
    formatoId = (cursor.execute("SELECT id FROM conf WHERE formato = ?", (formato,)).fetchone())[0]
    print(formatoId)
    meta = (cursor.execute("SELECT meta FROM conf WHERE formato = ?", (formato,)).fetchone())[0]
    print(meta)
    conf = (cursor.execute("SELECT confidence FROM conf WHERE formato = ?", (formato,)).fetchone())[0]
    print(conf)

    cbFormato.set(formato)
    slConf.set(conf)
    slValue.configure(text=conf)
    conf = conf/100


getDBdata()


def insertRegDB(piezas):
    global horaPlot, placas, factor, acumPiezas, metaPieza, topProm, promMeta, promTot

    print(formatoId)
    print(formato)
    print(meta)
    print(conf)
    now = datetime.datetime.now()
    print(now)
    mes = now.month
    dia = now.day
    hora = now.hour

    turno = getTurno(now)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++InsertDB1")
    #Acumular placas por hora
    actPlacas = placas[hora] + 1
    placas[hora] = actPlacas

    #Calcular piezas meta actual
    metaPieza[hora] = actPlacas * meta

    #Acumular piezas actuales
    actPiezas = acumPiezas[hora] + piezas
    acumPiezas[hora] = actPiezas

    print(actPiezas)
    print(type(actPiezas))
    print(metaPieza[hora])
    print(type(metaPieza[hora]))
          
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++InsertDB2")

    #Calcular promedio por hora
    if not metaPieza[hora] == 0:
        factor[hora] = round((actPiezas/metaPieza[hora]) * 100)
    
    print(placas)
    print(metaPieza)
    print(actPiezas)
    print(factor)


    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++InsertDB3")

    #Calcular promedio de ultimas 10
    promMeta.append(meta)
    promTot.append(piezas)
    print("++++++++++++++++++++++++++++++++++++++++++++++++lista de metas")
    print(promMeta)
    print(promTot)
    if len(promMeta) >= 11:
        promMeta.pop(0)
        promTot.pop(0)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++InsertDB4")




    # query = "INSERT INTO data " \
    #         "(id, formato,piezasMeta, piezasReal, conf, datetime, mes, dia, hora, turno) VALUES" \
    #         "( ?,?,?,?,?,?,?,?,?,?)"
    # param = (formatoId, formato, meta, piezas, conf, now, mes, dia, hora, turno)
    #
    # cursor.execute(query, param)
    # connection.commit()




def formatoCambiado(event):
    global formato, formatoId, conf, meta

    nuevoFormato = cbFormato.get()
    print("Nuevo formato: %s" % nuevoFormato)
    query = "UPDATE run SET formato = ?"
    cursor.execute(query, (nuevoFormato,))
    connection.commit()
    getDBdata()


cbFormato.bind('<<ComboboxSelected>>', formatoCambiado)

def sliderChange(event):
    global formato, conf
    conf = round(float(event))
    slValue.configure(text=conf)

    query = "UPDATE conf SET confidence = ? where formato = ?"
    cursor.execute(query, (conf,formato,))
    connection.commit()
    conf = conf/100
    print(conf)

slConf.configure(command=sliderChange)

##################################################################  Indicadores

frameIndicadores = ttk.Frame(style="success.TFrame")
frameIndicadores.pack(side='top', fill='both')


efFactor = ttk.Meter(frameIndicadores, metersize=200,
    padding=5,
    amountused=0,
    amounttotal=100,
    metertype="semi",
    subtext="Factor entrada",
    interactive=False,)
efFactor.pack(side='left')


#semaforo = ttk.Frame(frameIndicadores, style='primary')
#semaforo.pack(side='left', fill='x')


############################################################# Imagenes

frameImages = ttk.Frame()
frameImages.pack(side='top')

try:
    im = Image.open("opencv_frame_21.png")
    im = im.resize((600, 400))
except:
    im = im = Image.open("loading.jpg")
    im = im.resize((600, 400))
photoOrig = ImageTk.PhotoImage(im)


#vidPlaca = ttk.Label(frameImages, image=photoOrig)
vidPlaca = ttk.Label(frameImages, image=photoOrig)
vidPlaca.pack(side='left', padx = 100)

imgPiezas = ttk.Label(frameImages, image=photoOrig)
imgPiezas.pack(side='left', padx = 100)

############################################### Logs


frameLogs = ttk.Frame()
frameLogs.pack(side='top',fill = 'both')


##################################################### funciones DB
def insertPlacaDb(piezas):
    pass
    




##################################################### detect
def detect(file):
    global  modelPieza, conf
    device = 0
    if device != "cpu":
        try:
            print("Pasando imagen a modelPIeza_predict...")
            resultado = modelPieza.predict(file, stream=True, iou=0.2, conf=conf, device=device, half=True)
        except:
            print("Error en modelPieza_predict")
    else:
        resultado = modelPieza.predict(file, stream=True, iou=0.2, conf=conf, device=device)

    for res in resultado:
        piezasImg = res.plot(labels=False)

        n = len(res)
        try:
            #if not disableDB:
            print("Insertando en DB")
            insertRegDB(n)
            pass
        except:
            print("Error al insertar db")
            
        factor = round(n / meta * 100, 1)
        #piezasImg = cv2.resize(piezasImg, (480, 480))

        try:
            print("agregando texto a imagen modelPIeza_predict")
            cv2.putText(piezasImg, f"{n} {formato} {factor}{'%'}", org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 255, 0), thickness=3)
        except:
            print("Error agregando texto a imagen en modelPieza_predict")


        try:
            print("seteando imagen en imgPiezas...")
            piezasImg = cv2.cvtColor(piezasImg, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(piezasImg)
            image = ImageTk.PhotoImage(image=frame)
            imgPiezas.configure(image=image)
            imgPiezas.image = image
        except:
            print("Error seteando imagen en imgPiezas")
        #cv2.imwrite(save_path, im0)
        #cv2.imwrite(os.path.join(crop_dir_name, "100.png"), piezasImg)







############################################## Class Object Counter
class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""
    global im88, lastTrackId
    def __init__(self):
        """Initializes the Counter with default values for various tracking and counting parameters."""

        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = [(20, 400), (1260, 400)]
        self.line_dist_thresh = 15
        self.counting_region = None
        self.region_color = (255, 0, 255)
        self.region_thickness = 5

        # Image and annotation Information
        self.im0 = None
        self.tf = None
        self.view_img = False
        self.view_in_counts = True
        self.view_out_counts = True

        self.names = None  # Classes names
        self.annotator = None  # Annotator

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.counting_list = []
        self.count_txt_thickness = 0
        self.count_txt_color = (0, 0, 0)
        self.count_color = (255, 255, 255)

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False
        self.track_color = (0, 255, 0)

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        classes_names,
        reg_pts,
        count_reg_color=(255, 0, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=False,
        count_txt_thickness=2,
        count_txt_color=(0, 0, 0),
        count_color=(255, 255, 255),
        track_color=(0, 255, 0),
        region_thickness=5,
        line_dist_thresh=15,
    ):
        """
        Configures the Counter's image, bounding box line thickness, and counting region points.

        Args:
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): Flag to control whether to display the video stream.
            view_in_counts (bool): Flag to control whether to display the incounts on video stream.
            view_out_counts (bool): Flag to control whether to display the outcounts on video stream.
            reg_pts (list): Initial list of points defining the counting region.
            classes_names (dict): Classes names
            track_thickness (int): Track thickness
            draw_tracks (Bool): draw tracks
            count_txt_thickness (int): Text thickness for object counting display
            count_txt_color (RGB color): count text color value
            count_color (RGB color): count text background color value
            count_reg_color (RGB color): Color of object counting region
            track_color (RGB color): color for tracks
            region_thickness (int): Object counting Region thickness
            line_dist_thresh (int): Euclidean Distance threshold for line counter
        """
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks

        # Region and line selection
        if len(reg_pts) == 2:
            print("Line Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = LineString(self.reg_pts)
        elif len(reg_pts) == 4:
            print("Region Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = Polygon(self.reg_pts)
        else:
            print("Invalid Region points provided, region_points can be 2 or 4")
            print("Using Line Counter Now")
            self.counting_region = LineString(self.reg_pts)

        self.names = classes_names
        self.track_color = track_color
        self.count_txt_thickness = count_txt_thickness
        self.count_txt_color = count_txt_color
        self.count_color = count_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.line_dist_thresh = line_dist_thresh

    def mouse_event_for_region(self, event, x, y, flags, params):
        """
        This function is designed to move region with mouse events in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any flags associated with the event (e.g., cv2.EVENT_FLAG_CTRLKEY,
                cv2.EVENT_FLAG_SHIFTKEY, etc.).
            params (dict): Additional parameters you may want to pass to the function.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts):
                if (
                    isinstance(point, (tuple, list))
                    and len(point) >= 2
                    and (abs(x - point[0]) < 10 and abs(y - point[1]) < 10)
                ):
                    self.selected_point = i
                    self.is_drawing = True
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                self.counting_region = Polygon(self.reg_pts)

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.selected_point = None

    def extract_and_process_tracks(self, tracks):
        global lastTrackId
        """Extracts and processes tracks for object counting in a video stream."""
        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        track_ids = tracks[0].boxes.id.int().cpu().tolist()

        # Annotator Init and region drawing

        self.annotator = Annotator(self.im0, self.tf, self.names)
        self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)

        # Extract tracks
        for box, track_id, cls in zip(boxes, track_ids, clss):
            # Draw bounding box
            self.annotator.box_label(box, label=f"{track_id}:{self.names[cls]}", color=colors(int(cls), True))

            # Draw Tracks
            track_line = self.track_history[track_id]
            track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
            if len(track_line) > 30:
                track_line.pop(0)

            # Draw track trails
            if self.draw_tracks:
                self.annotator.draw_centroid_and_tracks(
                    track_line, color=self.track_color, track_thickness=self.track_thickness
                )

            prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

            # Count objects
            if len(self.reg_pts) == 4:
                if (
                    prev_position is not None
                    and self.counting_region.contains(Point(track_line[-1]))
                    and track_id not in self.counting_list
                    and lastTrackId != track_id
                ):
                    self.counting_list.append(track_id)



                    if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                        self.in_counts += 1
                        crop_obj = im88[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                        #cv2.imwrite(os.path.join(crop_dir_name, str(track_id) + ".png"), crop_obj)
                        print("++++++++++++++++++++aqui")
                        detect(crop_obj)
                    else:
                        self.out_counts += 1
                        crop_obj = im88[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                        #cv2.imwrite(os.path.join(crop_dir_name, str(track_id) + ".png"), crop_obj)
                        print("++++++++++++++++++++aqui")
                        #
                        detect(crop_obj)
                    lastTrackId = track_id

            elif len(self.reg_pts) == 2:
                if prev_position is not None:
                    distance = Point(track_line[-1]).distance(self.counting_region)



                    if distance < self.line_dist_thresh and track_id not in self.counting_list and lastTrackId != track_id:
                        self.counting_list.append(track_id)
                        if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                            self.in_counts += 1
                            crop_obj = im88[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                            #cv2.imwrite(os.path.join(crop_dir_name, str(track_id) + ".png"), crop_obj)
                            detect(crop_obj)
                            print("------------------------aca")
                        else:
                            self.out_counts += 1
                            crop_obj = im88[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                            #cv2.imwrite(os.path.join(crop_dir_name, str(track_id) + ".png"), crop_obj)
                            print("------------------------aca")
                            detect(crop_obj)
                        lastTrackId = track_id
        incount_label = f"In Count : {self.in_counts}"
        outcount_label = f"OutCount : {self.out_counts}"

        # Display counts based on user choice
        counts_label = None
        if not self.view_in_counts and not self.view_out_counts:
            counts_label = None
        elif not self.view_in_counts:
            counts_label = outcount_label
        elif not self.view_out_counts:
            counts_label = incount_label
        else:
            counts_label = f"{incount_label} {outcount_label}"

        if counts_label is not None:
            self.annotator.count_labels(
                counts=counts_label,
                count_txt_size=self.count_txt_thickness,
                txt_color=self.count_txt_color,
                color=self.count_color,
            )
        self.counting_list = []
        

    def display_frames(self):
        """Display frame."""
        if self.env_check:
            #cv2.namedWindow("Ultralytics YOLOv8 Object Counter")
            if len(self.reg_pts) == 4:  # only add mouse event If user drawn region
                cv2.setMouseCallback(
                    "Ultralytics YOLOv8 Object Counter", self.mouse_event_for_region, {"region_points": self.reg_pts}
                )
            #cv2.imshow("Ultralytics YOLOv8 Object Counter", self.im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image


        if tracks.boxes.id is None:
            if self.view_img:
                self.display_frames()
            return im0
        self.extract_and_process_tracks(tracks)

        if self.view_img:
            self.display_frames()
        return self.im0


############################################## Declare Init counter
model = YOLO("weights/best_s.pt")
#model.to('cuda')
region_points = [(250, 20),(350, 20),(350, 1200), (250, 1200)]
counter = ObjectCounter()
counter.set_args(view_img=False,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True)

crop_dir_name = "test"
if not os.path.exists(crop_dir_name):
    os.mkdir(crop_dir_name)


################################################ Get video
if not devWindows:
    #from pygigev import PyGigEV as gev
    import cv2
    import numpy as np

    #gst = 'tdgigevsrc cam-index=0 ! bayer2rgb ! videoconvert ! video/x-raw, format=BGR ! appsink' #COLOR ok

#gst = 'tdgigevsrc cam-index=0 ! bayer2rgb ! appsink'
#cap= cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

    #cap= cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    #os.system('./loadfeatures param.txt')
    # create new context to store native camera data
    #ctx = gev()

    # print list of available cameras
    #print(ctx.GevGetCameraList())
    #while True:
    #    cams = ctx.GevGetCameraList()
    #    print(cams[0])
    #    if cams[0] == 'OK':
    #        break
    #    else:
    #        print("Esperando conexion de camara...")
##
##    # open the first detected camera - returns 'OK'
##    ctx.GevOpenCamera()
#    gst = 'tdgigevsrc cam-index=0 nfrm=8 ! bayer2rgb ! videoconvert ! video/x-raw, format=BGR, framerate=30/1 ! appsink sink=false max-buffers=5 drop=True' #COLOR ok
    #gst = 'gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1, format=YUY2 ! videocrop name=cropper ! videoscale ! video/x-raw, width=640, height=480 ! videoconvert  ! appsink sync=False'
    gst = 'gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1, format=YUY2 ! videocrop name=cropper ! videoscale ! video/x-raw, width=640, height=480 ! videoconvert  ! appsink sync=False'
    
    cap= cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    #cap= cv2.VideoCapture("video.mp4")
##    # get image parameters - returns python object of params
##    params = ctx.GevGetImageParameters()
##    print("Initial image parameters:")
##    print(params)
##
##    # camera sensor properties
##    width_max = 1936
##    height_max = 1216
##    binning = 0
##    saturation = 0
##    brightness = 0
##    contrast = 0
##
##    # desired properties
##    crop_factor = 4.0
##    width = int(width_max * 1/crop_factor)
##    height = int(height_max * 1/crop_factor)
##    x_offset = int((width_max - width) / 2)
##    y_offset = int((height_max - height) / 2)
##
##    ctx.GevSetImageParameters(width,
##                              height,
##                              x_offset,
##                              y_offset,
##                              params['pixelFormat'][0])
##    params = ctx.GevGetImageParameters()
##    print("Final image parameters:")
##    print(params)
##
##    width = params['width']
##    height = params['height']
##
##    # allocate image buffers and prepare for async image transfer to buffer
##    ctx.GevInitializeImageTransfer(1)
##
##    # start transfering images to memory buffer,
##    # use -1 for streaming or [1-9] for num frames
##    ctx.GevStartImageTransfer(-1)


def getVideoPlaca():
    global im88
    region_points = [(500, 20), (500, 1260)]
    try:
        print("Leyendo imagen de stream")
        ret,frame = cap.read()
        if frame is None:
        	print("Error leyendo frame!!!!!!")
    except:
        print("Error al leer imagen de stream")
    #try:  COLOR_BGR2RGB
##    im0 = ctx.GevGetImageBuffer().reshape(height, width)
##    im88 = ctx.GevGetImageBuffer().reshape(height, width)
    im0 = frame
    im88 = frame
    #im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    im0 = cv2.resize(im0, (640, 480))
    #im88 = cv2.cvtColor(im88, cv2.COLOR_BAYER_BG2BGR)
    im88 = cv2.resize(im88, (640, 480))
    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    try:
        print("Pasando imagen a track...")
        tracks = model.track(im0, conf=0.3,stream=True, persist=True, show=False)
    except:
        print("Error procesando track")
  
    for track in tracks:
        try:
            print("Pasando imagen a count...")
            newTrack = track
            newImg = counter.start_counting(im0, newTrack)
            newImg = track.plot()
        except:
            print("Error en count")
        try:
            print("Seteando imagen en vidPlaca")
            frame = Image.fromarray(newImg)
            image = ImageTk.PhotoImage(image=frame)
            vidPlaca.configure(image=image)
            vidPlaca.image = image
        except:
            print("Error seteando imagen en vidPlaca")
    
##    frame = Image.fromarray(im0)
##    image = ImageTk.PhotoImage(image=frame)
##    vidPlaca.configure(image=image)
##    vidPlaca.image = image
    
    vidPlaca.after(1, getVideoPlaca)




if not devWindows:
    getVideoPlaca()




def updateFactor():
    global promedio
    factorProm = promedio
    efFactor.configure(amountused=factorProm)
    if factorProm >= 98:
        frameIndicadores.configure(style='success.TFrame')
        efFactor.configure(style='success.TFrame')

    elif factorProm < 98 and factorProm >= 92:
        frameIndicadores.configure(style='warning.TFrame')
        efFactor.configure(style='warning.TFrame')

    elif factorProm < 92:
        frameIndicadores.configure(style='danger.TFrame')
        efFactor.configure(style='danger.TFrame')
    root.after(1000, updateFactor)


global u
u = 0
canvas = None

def plot():
    global u, canvas
    global horaPlot, placas, factor
    #hora = []
    #placas = []
    #factor = []


    
    if canvas:
        canvas.get_tk_widget().pack_forget()
        canvas.get_tk_widget().destroy()
        canvas.figure.clear()
        canvas.figure.clf()
    """
    if ax:
        ax.clear()
    """

    #hora = [0,1,2,3,4,5,6]
    #factor = [89,99,34,90,96,96,u+81]

    f = Figure(figsize=(7, 3), dpi=100, facecolor="black")
    ax = f.add_subplot(111)

    ax.set_xlabel('Hora')
    ax.xaxis.label.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.tick_params(axis='x', colors = 'white')

    ax.set_ylabel('% Factor')
    ax.yaxis.label.set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='y', colors='white')


    width = 0.9

    """
    for horas in hora:
        p =  ax.bar(hora, factor, width)
        ax.bar_label(p,label_type='center')
    """
    bar_container = ax.bar(horaPlot, factor, width)
    
    # Start mod Ab 12-23-24 10:13:00.00
    # meant to plot zeroes allowing tkinter visualization when factor list is all zeroes
    
    # ax.bar_label(bar_container, fmt='{:.0f}%' ,label_type='center')
    
    if all(value == 0 for value in bar_container.datavalues):
    	print('bar_container is all zeroes')
    	ax.bar_label(bar_container, fmt='0', label_type='center')
    else:
    	print('_____________________________________________________________________________________________________________________________________________________________________________________________')
    	print(f'bar_container: {bar_container}')
    	ax.bar_label(bar_container, fmt='{:.0f}%', label_type='center')
    	
    # End mod Ab 12-23-24 10:13:00.00


    if canvas:
        canvas.get_tk_widget().pack_forget()
        
    canvas = FigureCanvasTkAgg(f, master=frameLogs)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

    u = u + 1
    #ax.clear()
    #canvas.get_tk_widget().pack_forget()
    #canvas.figure.clear()
    #canvas.figure.clf()
    #canvas.get_tk_widget().destroy()
    

    #ax.clear()
    gc.collect()
    print(u)
    root.after(20*1000, plot)


def getProm():

    global promedio, formato, topProm, promMeta, promTot

    # query = "SELECT formato, sum(piezasMeta), sum(piezasReal)" \
    #         "FROM (SELECT formato, piezasMeta, piezasReal, row_number() OVER (PARTITION BY formato ORDER BY datetime  DESC   ) AS RowNum " \
    #         "FROM data) " \
    #         "WHERE RowNum <= 10 " \
    #         " AND formato = \"" + formato + "\" GROUP BY formato"
    try:
        #print(query)
        #cursor.execute(query)
        #rows = cursor.fetchall()
        #sumMeta = rows[0][1]
        #sumReal = rows[0][2]
        #print(rows)
        print("calculando promedio")
        print(promTot)
        print(promMeta)
        if (len(promTot)) >= 10:
            print("Calculando promedio de 10 placas")
            sumMeta = sum(promMeta)
            sumReal = sum(promTot)
            print(sumMeta)
            print(sumReal)
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Promedio: " + formato )
            prom = (sumReal / sumMeta) * 100
            promedio = round(prom, 1)
    except:
        print("Error calculando promedio")

def queryData():
    global horaPlot, placas, factor, acumPiezas, metaPieza
             
    # queryold = "SELECT hora, count(piezasMeta) as placas, round(((piezasReal * 1.0 / piezasMeta * 1.0)*100),1) as factor " \
    #         "FROM ( SELECT hora, piezasMeta, piezasReal, row_number() OVER (PARTITION BY hora ORDER BY datetime  DESC ) AS RowNum FROM data WHERE datetime > date() ) " \
    #         "GROUP BY hora"
    #
    # query = "SELECT hora, placas, round(((piezasReal * 1.0 / piezasMeta * 1.0)*100),1) as factor FROM (SELECT hora, count(piezasMeta) as placas,sum(piezasMeta) as piezasMeta, sum(piezasReal) as piezasReal from data WHERE datetime >= date() GROUP BY hora)"
    #
    while True:
        horaIniTurno = datetime.time(0,0,0)
        horaIniTurno2 = datetime.time(0,0,30)
        horaAct = datetime.datetime.now().time()

        if (horaAct >= horaIniTurno) and (horaAct <= horaIniTurno2):
            horaPlot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24]
            placas = [0] * 24
            factor = [0] * 24
            acumPiezas = [0] * 24
            metaPieza = [0] * 24
        #try:
            # horaPlot = []
            # placas = []
            # factor = []
              
            # cursor.execute(query)
            # rows = cursor.fetchall()
            # print(rows)

            # for item in rows:
            #     horaPlot.append(item[0])
            #     placas.append(item[1])
            #     factor.append(item[2])
            #     print(horaPlot)
            #     print(placas)
            #     print(factor)
        #except:
        #    print("Error leyendo datos de db")
            
        getProm()
        time.sleep(10)
        
#if not disableDB:
t = threading.Thread(target=queryData)
t.start()
plot()
updateFactor()



root.mainloop()
