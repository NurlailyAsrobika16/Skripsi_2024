import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.uic import loadUi
from tensorflow.lite.python.interpreter import Interpreter
from PyQt5 import QtCore, QtGui, QtWidgets

class QThreadVideo(QtCore.QThread):
    output_frame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super(QThreadVideo, self).__init__()
        self.modelpath = 'detect.tflite'
        self.interpreter = Interpreter(model_path=self.modelpath)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.float_input = (self.input_details[0]['dtype'] == np.float32)
        self.input_mean = 127.5
        self.input_std = 127.5

    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Upload Video", "", "Video Files (*.mp4)")
        if file_path:
            self.detect_objects(file_path)

    def detect_objects(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Ubah warna frame dari BGR menjadi RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imH, imW, _ = frame.shape
            image_resized = cv2.resize(image_rgb, (self.width, self.height))
            input_data = np.expand_dims(image_resized, axis=0)

            # Normalisasi nilai piksel jika menggunakan model floating (non-quantized)
            if self.float_input:
                input_data = (np.float32(input_data) - self.input_mean) / self.input_std

            # Lakukan deteksi objek dengan menjalankan model TFLite
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            boxes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            # Tampilkan hasil deteksi objek pada layar
            for i in range(len(scores)):
                if scores[i] > 0.5:  # Ubah threshold sesuai dengan kebutuhan Anda
                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                    label = f"Grade: {int(classes[i])}, Score: {scores[i]}"
                    cv2.putText(frame, label, (xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Emit frame yang sudah diolah
            self.output_frame.emit(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(430, 589)
        MainWindow.setMinimumSize(430, 589)  # Set ukuran minimum jendela
        MainWindow.setMaximumSize(430, 589) 
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("background-color: #419ca6")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.screen = QtWidgets.QLabel(self.centralwidget)
        self.screen.setGeometry(QtCore.QRect(50, 120, 321, 231))
        self.screen.setStyleSheet("background-color:#ffffff;border: 3px ; border-radius:5px;")
        self.screen.setText("")
        self.screen.setObjectName("screen")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(50, 60, 171, 21))
        self.label_8.setStyleSheet("background-color:transparent;color:#ffffff;")
        self.label_8.setObjectName("label_8")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(200, 30, 161, 31))
        self.title.setMaximumSize(QtCore.QSize(16777211, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.title.setFont(font)
        self.title.setStyleSheet("background-color:transparent;color:#ffffff;")
        self.title.setTextFormat(QtCore.Qt.AutoText)
        self.title.setObjectName("title")
        self.uploadVideo = QtWidgets.QPushButton(self.centralwidget)
        self.uploadVideo.setGeometry(QtCore.QRect(60, 390, 91, 41))
        self.uploadVideo.setStyleSheet("QPushButton {   background:#0b4f62;\n"
"                                  border: 1px solid rgb(215,215,215);\n"
"                                   border-radius:5px;\n"
"                                   color:rgb(255,255,255);\n"
"                                  }\n"
"                                  \n"
"                                  QPushButton:hover{\n"
"                                color:rgba(20,20,20,235);\n"
"                                color:rgb(255,255,255);\n"
"                                 border:none;\n"
"                                }\n"
"                            ")
        self.uploadVideo.setObjectName("uploadVideo")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(210, 390, 141, 16))
        self.label_3.setStyleSheet("color:#ffffff;background-color:transparent")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(210, 420, 91, 16))
        self.label_4.setStyleSheet("color:#ffffff;background-color:transparent")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(320, 420, 51, 16))
        self.label_5.setStyleSheet("color:#ffffff;background-color:transparent")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(210, 440, 31, 16))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(320, 440, 31, 16))
        self.label_7.setObjectName("label_7")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(210, 470, 151, 16))
        self.label_9.setStyleSheet("color:#ffffff;background-color:transparent")
        self.label_9.setObjectName("label_9")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(210, 490, 113, 20))
        self.lineEdit.setStyleSheet("QLineEdit {\n"
"                                        background:#ffffff;\n"
"                                    border:2px#403e52;\n"
"                                     border-radius: 5px;\n"
"                                    color:#403e52;\n"
"                                     padding-left:10px;\n"
"                                    }")
        self.lineEdit.setObjectName("lineEdit")
        self.ok = QtWidgets.QPushButton(self.centralwidget)
        self.ok.setGeometry(QtCore.QRect(330, 490, 41, 23))
        self.ok.setStyleSheet("QPushButton {\n"
"                              background:#0b4f62;\n"
"                              border: 1px solid rgb(215,215,215);\n"
"                              border-radius:5px;\n"
"                              color:rgb(255,255,255);\n"
"                              }\n"
"                            \n"
"                              QPushButton:hover{\n"
"                              color:rgba(20,20,20,235);\n"
"                              color:rgb(255,255,255);\n"
"                              border:none;\n"
"                              }\n"
"                              ")
        self.ok.setObjectName("ok")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(210, 520, 47, 13))
        self.label_10.setStyleSheet("background-color:transparent;color:#ffffff;")
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(20, 10, 381, 571))
        self.label_11.setStyleSheet("background-color:#265d62;border: 3px ; border-radius:7px;")
        self.label_11.setText("")
        self.label_11.setObjectName("label_11")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(170, 10, 241, 571))
        self.label_18.setStyleSheet("background:#30777d;border: 3px ; border-radius:7px;")
        self.label_18.setObjectName("label_18")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(210, 410, 118, 3))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_11.raise_()
        self.label_18.raise_()
        self.screen.raise_()
        self.label_8.raise_()
        self.title.raise_()
        self.uploadVideo.raise_()
        self.label_3.raise_()
        self.label_4.raise_()
        self.label_5.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.label_9.raise_()
        self.lineEdit.raise_()
        self.ok.raise_()
        self.label_10.raise_()
        self.line.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def configure(self, main_window):
        self.video = QThreadVideo()
        self.video.output_frame.connect(self.display_frame)  # Menghubungkan sinyal dengan metode display_frame
        self.uploadVideo.clicked.connect(self.video.upload_video)

    def display_frame(self, frame):
        # Menampilkan frame di QLabel screen
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.screen.setPixmap(pixmap)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_8.setText(_translate("MainWindow", "Masukkan Data Video:"))
        self.title.setText(_translate("MainWindow", "   Aplikasi Penghitung \n"
"Bibit Ikan Lele Otomatis"))
        self.uploadVideo.setText(_translate("MainWindow", "Upload Video"))
        self.label_3.setText(_translate("MainWindow", "<strong>Jumlah Lele Terdeteksi :</strong>"))
        self.label_4.setText(_translate("MainWindow", "Data Sebenarnya"))
        self.label_5.setText(_translate("MainWindow", "Data Total"))
        self.label_6.setText(_translate("MainWindow", "0"))
        self.label_7.setText(_translate("MainWindow", "0"))
        self.label_9.setText(_translate("MainWindow", "<strong>Jumlah Lele Sebenarnya :</strong>"))
        self.lineEdit.setText(_translate("MainWindow", "0"))
        self.ok.setText(_translate("MainWindow", "OK"))
        self.label_10.setText(_translate("MainWindow", "Akurasi :"))
        self.label_18.setText(_translate("MainWindow", "TextLabel"))

        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.configure(MainWindow) 
    MainWindow.show()
    sys.exit(app.exec_())