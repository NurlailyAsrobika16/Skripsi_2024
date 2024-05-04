import sensor, image, time, tf, pyb

# Inisialisasi kamera OpenMV
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_auto_gain(False) # must be turned off for color tracking
sensor.set_auto_whitebal(False) # must be turned off for color tracking
sensor.skip_frames(time=2000)  # Wait for sensor to adjust

# Inisialisasi model Keras
tfmodel = tf.load('model_lele11_fit_128_datagenerator_50epoch (1).tflite',load_to_fb=True)

# Threshold yang digunakan untuk find_blobs
thresholds = [(34, 100, -128, 127, -128, 127)]

# Get the class labels
class_labels = ['Grade A', 'Grade B', 'Grade C']

# Inisialisasi variabel perhitungan kelas
count_grade_a = 0
count_grade_b = 0
count_grade_c = 0

# Inisialisasi variabel untuk melacak apakah objek terdeteksi
objek_terdeteksi = False

# Inisialisasi variabel untuk perhitungan jumlah bibit lele
total_bibit_lele = 0

# Loop utama
clock = time.clock()
while (True):
    clock.tick()

    img = sensor.snapshot()

    # Find blobs
    blobs = img.find_blobs(thresholds,pixels_threshold=200, area_threshold=200, invert=True)

    # Reset nilai variabel jumlah grade jika tidak ada objek terdeteksi
    count_grade_a = 0
    count_grade_b = 0
    count_grade_c = 0

    for blob in blobs:
        # Calculate square size based on the maximum of width and height of the blob
        square_size = max(blob.w(), blob.h())

        # Calculate the top-left corner coordinates of the square
        x = blob.cx() - square_size // 2
        y = blob.cy() - square_size // 2

        # Increase the width of the square by multiplying the square_size with a factor
        width_factor = 1.2  # Adjust this factor according to your desired width increase
        square_size = int(square_size * width_factor)

        roi = [x,y,square_size, square_size]

        roi_img = img.copy(roi=roi)

        pyb.delay(1000) #delay untuk menangkap gambar
        for obj in tfmodel.classify(roi_img):
            # Get the predicted class label and confidence
            predicted_class = obj.output()
            max_result_value = max(predicted_class)
            most_likely_idx = predicted_class.index(max_result_value)

            # Get the predicted class label
            predicted_label = class_labels[most_likely_idx]

            # Get the confidence score
            confidence_score = max_result_value

            # Update counts based on the predicted class
            if most_likely_idx == 0:
                if max_result_value >= 0.5:
                    count_grade_a += 1
            elif most_likely_idx == 1:
                if max_result_value >= 0.5:
                    count_grade_b += 1
            elif most_likely_idx == 2:
                if max_result_value >= 0.5:
                    count_grade_c += 1

            print("-----------------------")
            # Print the predicted class label and confidence score
            print("Predicted Label: %s" % predicted_label)
            print("Confidence Score: %.2f" % confidence_score)

        # Draw a rectangle and a cross on the original image
        img.draw_rectangle((x, y, square_size,square_size), color=(0, 255, 0))
        img.draw_cross(blob.cx(), blob.cy(), color=(0, 255, 0))

        # Set objek_terdeteksi menjadi True karena telah terdeteksi objek
        objek_terdeteksi = True

    # Perbarui nilai total_bibit_lele jika objek terdeteksi
    if objek_terdeteksi:
        total_bibit_lele = count_grade_a + count_grade_b + count_grade_c

    img.draw_string(10, 10, "A: %d" % count_grade_a, color=(0, 0, 0))
    img.draw_string(10, 30, "B: %d" % count_grade_b, color=(0, 0, 0))
    img.draw_string(10, 50, "C: %d" % count_grade_c, color=(0, 0, 0))
    img.draw_string(10, 70, "Total: %d" % total_bibit_lele, color=(0, 0, 255))


    # Print FPS
    print(clock.fps())
