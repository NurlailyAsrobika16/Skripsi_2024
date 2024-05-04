import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.uic import loadUi
from tensorflow.lite.python.interpreter import Interpreter
from PyQt5 import QtCore, QtGui, QtWidgets

class QThreadVideo(QtCore.QThread):
    output_frame = QtCore.pyqtSignal(np.ndarray)
    total_counts_changed = QtCore.pyqtSignal(int, int, int, int)  # Emit sinyal untuk total counts

    def __init__(self):
        super(QThreadVideo, self).__init__()
        self.modelpath = 'model_lele11_fit_128_datagenerator_50epoch.tflite'
        self.interpreter = Interpreter(model_path=self.modelpath)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.float_input = (self.input_details[0]['dtype'] == np.float32)
        self.input_mean = 127.5
        self.input_std = 127.5
        
        # Default thresholds for blob detection
        self.default_thresholds = [(34, 100, -128, 127, -128, 127)]
        self.thresholds = self.default_thresholds.copy()

    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Upload Video", "", "Video Files (*.mp4)")
        if file_path:
            self.detect_objects(file_path)

    # Function to update thresholds
    def update_thresholds(self):
        # Add your logic here to update thresholds based on user input or any other criteria
        pass

    # Function to handle keyboard events
    def handle_keyboard_event(self, event):
        if event == ord('r'):  # Reset thresholds to default
            self.thresholds = self.default_thresholds.copy()
            print("Thresholds reset to default values.")
        elif event == ord('u'):  # Update thresholds
            self.update_thresholds()
            print("Thresholds updated.")
        # Add more key-event handling as needed

    def detect_objects(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, self.thresholds[0][0], self.thresholds[0][1], self.thresholds[0][2])
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            count_grade_a = 0
            count_grade_b = 0
            count_grade_c = 0

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                roi = frame[y:y + h, x:x + w]
                input_data = np.expand_dims(cv2.resize(roi, (self.width, self.height)), axis=0).astype(np.float32)
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

                max_result_idx = np.argmax(output_data)
                max_result_value = output_data[0][max_result_idx]
                if max_result_idx == 0:
                    if max_result_value >= 0.5:
                        count_grade_a += 1
                elif max_result_idx == 1:
                    if max_result_value >= 0.5:
                        count_grade_b += 1
                elif max_result_idx == 2:
                    if max_result_value >= 0.5:
                        count_grade_c += 1

            total_a = count_grade_a
            total_b = count_grade_b
            total_c = count_grade_c
            total_all = total_a + total_b + total_c

            self.total_counts_changed.emit(total_a, total_b, total_c, total_all)

            self.output_frame.emit(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(560, 589)
        MainWindow.setMaximumSize(QtCore.QSize(601, 689))
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("background-color: rgb(48, 122, 126)")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.screen = QtWidgets.QLabel(self.centralwidget)
        self.screen.setGeometry(QtCore.QRect(210, 80, 311, 221))
        self.screen.setStyleSheet("background-color:#ffffff;border: 3px ; border-radius:5px;")
        self.screen.setText("")
        self.screen.setObjectName("screen")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(210, 50, 171, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("background-color:transparent;color:#ffffff;")
        self.label_8.setObjectName("label_8")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(40, 50, 161, 61))
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
        self.uploadVideo.setGeometry(QtCore.QRect(60, 130, 91, 41))
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
        self.label_3.setGeometry(QtCore.QRect(220, 350, 151, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color:#ffffff;background-color:transparent")
        self.label_3.setObjectName("label_3")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(20, 10, 381, 571))
        self.label_11.setStyleSheet("background-color:#1b4347;border: 3px ; border-radius:7px;")
        self.label_11.setText("")
        self.label_11.setObjectName("label_11")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(190, 10, 351, 571))
        self.label_18.setStyleSheet("background:#23575b;border: 3px ; border-radius:7px;")
        self.label_18.setText("")
        self.label_18.setObjectName("label_18")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(30, 780, 744, 3))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.GradeA = QtWidgets.QLabel(self.centralwidget)
        self.GradeA.setGeometry(QtCore.QRect(210, 380, 201, 23))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(65)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.GradeA.sizePolicy().hasHeightForWidth())
        self.GradeA.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.GradeA.setFont(font)
        self.GradeA.setStyleSheet("background-color:#276568;color : rgb(255, 255, 255)")
        self.GradeA.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.GradeA.setObjectName("GradeA")
        self.TotalA = QtWidgets.QLabel(self.centralwidget)
        self.TotalA.setGeometry(QtCore.QRect(420, 380, 112, 23))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TotalA.sizePolicy().hasHeightForWidth())
        self.TotalA.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.TotalA.setFont(font)
        self.TotalA.setStyleSheet("background-color:#276568")
        self.TotalA.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.TotalA.setObjectName("TotalA")
        self.GradeB = QtWidgets.QLabel(self.centralwidget)
        self.GradeB.setGeometry(QtCore.QRect(210, 410, 201, 21))
        font.setBold(True)
        font.setWeight(65)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.GradeB.sizePolicy().hasHeightForWidth())
        self.GradeB.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.GradeB.setFont(font)
        self.GradeB.setStyleSheet("color : rgb(255, 255, 255);background-color:#276568")
        self.GradeB.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.GradeB.setObjectName("GradeB")
        self.GradeC = QtWidgets.QLabel(self.centralwidget)
        self.GradeC.setGeometry(QtCore.QRect(210, 440, 201, 21))
        font.setBold(True)
        font.setWeight(65)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.GradeC.sizePolicy().hasHeightForWidth())
        self.GradeC.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.GradeC.setFont(font)
        self.GradeC.setStyleSheet("color :#ffffff;background-color:#276568")
        self.GradeC.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.GradeC.setObjectName("GradeC")
        self.Total = QtWidgets.QLabel(self.centralwidget)
        self.Total.setGeometry(QtCore.QRect(210, 470, 201, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(65)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Total.sizePolicy().hasHeightForWidth())
        self.Total.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.Total.setFont(font)
        self.Total.setStyleSheet("color :#ffffff;background-color:#276568")
        self.Total.setObjectName("Total")
        self.TotalB = QtWidgets.QLabel(self.centralwidget)
        self.TotalB.setGeometry(QtCore.QRect(420, 410, 112, 23))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TotalB.sizePolicy().hasHeightForWidth())
        self.TotalB.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.TotalB.setFont(font)
        self.TotalB.setStyleSheet("background-color:#276568")
        self.TotalB.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.TotalB.setObjectName("TotalB")
        self.TotalC = QtWidgets.QLabel(self.centralwidget)
        self.TotalC.setGeometry(QtCore.QRect(420, 440, 112, 23))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TotalC.sizePolicy().hasHeightForWidth())
        self.TotalC.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.TotalC.setFont(font)
        self.TotalC.setStyleSheet("background-color:#276568")
        self.TotalC.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.TotalC.setObjectName("TotalC")
        self.Total_2 = QtWidgets.QLabel(self.centralwidget)
        self.Total_2.setGeometry(QtCore.QRect(420, 470, 112, 23))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Total_2.sizePolicy().hasHeightForWidth())
        self.Total_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Total_2.setFont(font)
        self.Total_2.setStyleSheet("background-color:#276568")
        self.Total_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.Total_2.setObjectName("Total_2")
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(240, 310, 251, 22))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.label_11.raise_()
        self.label_18.raise_()
        self.screen.raise_()
        self.label_8.raise_()
        self.title.raise_()
        self.uploadVideo.raise_()
        self.label_3.raise_()
        self.line_2.raise_()
        self.GradeA.raise_()
        self.TotalA.raise_()
        self.GradeB.raise_()
        self.GradeC.raise_()
        self.Total.raise_()
        self.TotalB.raise_()
        self.TotalC.raise_()
        self.Total_2.raise_()
        self.horizontalSlider.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    def configure(self, main_window):
        self.video = QThreadVideo()
        self.video.output_frame.connect(self.display_frame)
        self.video.total_counts_changed.connect(self.update_counts)  # Terima sinyal total counts
        self.uploadVideo.clicked.connect(self.video.upload_video)

    def update_counts(self, total_a, total_b, total_c, total_all):
        # Perbarui teks dari label-label yang sesuai
        self.TotalA.setStyleSheet("color: white")  # Mengatur warna putih dan posisi di tengah
        self.TotalA.setText(f"      {total_a}")
        
        self.TotalB.setStyleSheet("color: white")  # Mengatur warna putih dan posisi di tengah
        self.TotalB.setText(f"      {total_b}")
        
        self.TotalC.setStyleSheet("color: white")  # Mengatur warna putih dan posisi di tengah
        self.TotalC.setText(f"      {total_c}")
        
        self.Total_2.setStyleSheet("color: white")  # Mengatur warna putih dan posisi di tengah
        self.Total_2.setText(f"      {total_all}")
        
    def display_frame(self, frame):
        # Menampilkan frame di QLabel screen
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.screen.setPixmap(pixmap)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_8.setText(_translate("MainWindow", "Masukkan Data Video :"))
        self.title.setText(_translate("MainWindow", "Aplikasi Penghitung  \n"
"     Bibit Ikan Lele\n"
"        Otomatis"))
        self.uploadVideo.setText(_translate("MainWindow", "Upload Video"))
        self.label_3.setText(_translate("MainWindow", "Jumlah Lele Terdeteksi :"))
        self.GradeA.setText(_translate("MainWindow", "Grade A :"))
        self.GradeB.setText(_translate("MainWindow", "Grade B :"))
        self.GradeC.setText(_translate("MainWindow", "Grade C :"))
        self.Total.setText(_translate("MainWindow", "Total :"))
        
    def stop_detection(self):
        # Add logic to stop video detection
        pass

    def start_detection(self):
        # Add logic to start video detection
        pass
    


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.configure(MainWindow) 
    MainWindow.show()
    sys.exit(app.exec_())
