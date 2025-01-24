from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import io
from PIL import Image
import base64

cat_care_instructions = {
    'bengal': '''
        1. Berikan latihan fisik yang cukup untuk mengurangi energi berlebih, seperti mainan interaktif dan sesi bermain.
        2. Sediakan tempat tinggi untuk melompat dan bereksplorasi, seperti rak atau pohon kucing.
        3. Rutin menyikat bulu kucing Bengal untuk menjaga kebersihan dan kesehatan bulunya.
    ''',
    'british': '''
        1. Berikan kenyamanan dengan tempat tidur yang nyaman dan suasana yang tenang.
        2. Sediakan sesi bermain ringan meskipun mereka cenderung santai.
        3. Lakukan perawatan bulu secara rutin, meskipun pendek, untuk mengurangi rambut rontok.
    ''',
    'caracal': '''
        1. Sediakan ruang yang luas untuk berlari dan bermain agar mereka dapat mengeluarkan energi.
        2. Berikan makanan bergizi untuk mendukung kebutuhan energi mereka yang tinggi.
        3. Lakukan perawatan bulu secara rutin untuk menjaga kebersihannya.
    ''',
    'munchkin': '''
        1. Pastikan mereka memiliki mainan yang menarik dan aman untuk waktu bermain.
        2. Berikan perhatian ekstra dan waktu bermain yang cukup, karena mereka sangat aktif.
        3. Karena tubuh mereka yang kecil, pastikan mereka memiliki ruang yang aman untuk bergerak.
    ''',
    'persia': '''
        1. Sikat bulunya setiap hari untuk mencegah knot dan rambut rontok, karena bulunya yang panjang.
        2. Ciptakan lingkungan yang tenang dan nyaman untuk kucing Persia.
        3. Lakukan perawatan kebersihan secara rutin, terutama pada bulunya yang tebal.
    ''',
    'ragdoll': '''
        1. Pastikan mereka mendapatkan perhatian dan kasih sayang yang cukup dari pemiliknya.
        2. Sediakan tempat yang nyaman di dalam rumah karena mereka lebih suka berada di dalam.
        3. Rutin merawat bulunya meskipun tidak terlalu panjang, agar tetap bersih dan sehat.
    ''',
    'savannah': '''
        1. Berikan banyak stimulasi fisik dan mental, seperti mainan yang menantang dan sesi bermain interaktif.
        2. Pastikan mereka memiliki banyak ruang untuk bergerak dan bereksplorasi.
        3. Sediakan waktu untuk bermain aktif, karena mereka sangat cerdas dan aktif.
    ''',
    'sphynx': '''
        1. Mandikan kucing Sphynx secara teratur untuk menjaga kebersihan kulitnya yang rentan terhadap minyak.
        2. Pastikan mereka tetap hangat karena mereka tidak memiliki bulu untuk melindungi dari suhu dingin.
        3. Lakukan perawatan kulit secara rutin untuk mencegah penumpukan minyak dan menjaga kesehatannya.
    '''
}


# Inisialisasi Flask
app = Flask(__name__)

# Path model yang telah dilatih
model = load_model('model/model_terbaik.keras')  # Ganti dengan path model Anda

# Daftar class names (ganti sesuai dataset Anda)
class_names = ['bengal', 'british', 'caracal', 'munchkin', 'persia', 'ragdoll', 'savannah', 'sphynx'] # Ganti dengan nama kelas yang sesuai

# Folder untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Fungsi untuk memeriksa ekstensi file gambar
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Fungsi untuk memuat dan memproses gambar
def load_and_prepare_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Sesuaikan ukuran dengan model Anda
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Fungsi untuk memproses gambar dari data URL
def process_image_from_data_url(data_url):
    img_data = io.BytesIO(base64.b64decode(data_url.split(',')[1]))
    img = Image.open(img_data)
    img = img.resize((150, 150))  # Sesuaikan ukuran dengan model Anda
    img_array = np.array(img)

    # Pastikan array gambar memiliki tipe data float32 atau float64
    img_array = img_array.astype('float32')

    # Normalisasi (membagi dengan 255.0)
    img_array /= 255.0

    # Menambahkan dimensi batch (misalnya (1, 150, 150, 3))
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        # File gambar dari form upload
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img_array = load_and_prepare_image(filepath)
            # Prediksi menggunakan model
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_names[predicted_class_index]
            care_instructions = cat_care_instructions.get(predicted_class, "Deskripsi cara merawat tidak tersedia.")
            # Kembali ke halaman utama dengan hasil prediksi dan deskripsi cara merawat
            return render_template('index.html', filename=filename, predicted_class=predicted_class, care_instructions=care_instructions)

    elif 'file' in request.form:
        # Gambar dari kamera (data URL)
        img_data = request.form['file']
        img_array = process_image_from_data_url(img_data)

        # Prediksi menggunakan model
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        care_instructions = cat_care_instructions.get(predicted_class, "Deskripsi cara merawat tidak tersedia.")

        # Kirimkan response JSON untuk gambar dari kamera
        return jsonify(predicted_class=predicted_class, care_instructions=care_instructions)

    # Jika tidak ada file yang diterima, kembali ke halaman utama tanpa hasil
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
