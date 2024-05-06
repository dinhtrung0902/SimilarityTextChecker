import os
import pandas as pd
import threading
import time
import tkinter as tk
import torch
from tkinter import ttk, filedialog, messagebox
from googletrans import Translator
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from transformers import pipeline

# Tải pre-trained model và tokenizer
corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction")
model_name = "vinai/phobert-base-v2"
phobert = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Khởi tạo đối tượng Translator
translator = Translator()
# Hàm để chuyển đoạn văn thành vector
def get_vector(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = phobert(input_ids)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return vector
# Hàm để tạo vector cho mỗi câu trong đoạn văn bản
def get_sentence_vectors(text):
    # Tách câu từ đoạn văn bản
    sentences = text.split('.')
    sentence_vectors = []
    # Tạo vector cho mỗi câu
    for sentence in sentences:
        # Loại bỏ khoảng trắng ở đầu và cuối câu
        sentence = sentence.strip()
        if sentence:
            translated_sentence = translator.translate(sentence, src='auto', dest='vi').text
            vector = get_vector(translated_sentence)
            sentence_vectors.append(vector)
    return sentence_vectors
# Hàm tính cosine similarity giữa hai đoạn văn bản
def calculate_sentence_similarity(sentence_vectors1, sentence_vectors2):
    similarity_scores = []
    # Tính cosine similarity cho từng cặp câu từ hai đoạn văn bản
    for vector1 in sentence_vectors1:
        for vector2 in sentence_vectors2:
            similarity_score = cosine_similarity([vector1], [vector2])[0][0]
            if similarity_score < 0.8:
                similarity_score = 0
            similarity_scores.append(similarity_score)
    # Trả về điểm số tương đồng trung bình
    if similarity_scores:
        average_similarity = sum(similarity_scores) / len(similarity_scores)
    else:
        average_similarity = 0
    return average_similarity
# Hàm để tính tỉ lệ từ trùng lặp
def word_overlapping(text1, text2):
    # Chuyển đổi các chuỗi văn bản thành tập hợp các từ
    words1 = set(text1.split())
    words2 = set(text2.split())

    # Tính số lượng từ chung
    overlapping_words = len(words1.intersection(words2))

    # Tính tỉ lệ trùng lặp
    overlapping_ratio = overlapping_words / max(len(words1), len(words2))

    return overlapping_ratio
# Hàm để tính tỉ lệ trùng lặp số lượng từ, số lượng kí tự, số lượng câu
def text_similarity_percentage(text1, text2):
    # Tính số từ, số câu và số kí tự của mỗi văn bản
    num_words1 = len(text1.split())
    num_words2 = len(text2.split())
    num_sentences1 = text1.count('.') + 1
    num_sentences2 = text2.count('.') + 1
    num_chars1 = len(text1)
    num_chars2 = len(text2)

    # Tính tỉ lệ giữa số từ, số câu và số kí tự của hai văn bản
    word_ratio = min(num_words1, num_words2) / max(num_words1, num_words2)
    sentence_ratio = min(num_sentences1, num_sentences2) / max(num_sentences1, num_sentences2)
    char_ratio = min(num_chars1, num_chars2) / max(num_chars1, num_chars2)

    # Tính tỉ lệ trung bình
    similarity_percentage = (word_ratio + sentence_ratio + char_ratio) / 3

    return similarity_percentage

class App():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Kiểm tra đạo văn")
        self.root.geometry('1300x500')
        self.center_window()

        self.mainframe = tk.Frame(self.root, background="#f0f0f0")
        self.mainframe.pack(fill='both', expand=True, padx=20, pady=20)
        self.save_button = ttk.Button(self.mainframe, text="Lưu vào Excel", command=self.save_to_excel)
        self.save_button.grid(column=2, row=4, pady=5, sticky="w")
        self.text1 = ttk.Label(self.mainframe, text="Văn bản thứ nhất", background="#f0f0f0",
                              font=("Arial", 15))
        self.text1.grid(column=0, row=0, padx=10, pady=5, sticky="w")
        self.text2 = ttk.Label(self.mainframe, text="Văn bản thứ hai", background="#f0f0f0",
                              font=("Arial", 15))
        self.text2.grid(column=0, row=2, padx=10, pady=5, sticky="w")

        self.set_text_field1 = tk.Text(self.mainframe, font=("Arial", 12), height=3, wrap="word", width=50)
        self.set_text_field1.grid(column=1, row=0, pady=5, sticky="we")
        self.set_text_field2 = tk.Text(self.mainframe, font=("Arial", 12), height=3, wrap="word", width=50)
        self.set_text_field2.grid(column=1, row=2, pady=5, sticky="we")

        self.open_file_button1 = ttk.Button(self.mainframe, text="Mở File thứ nhất",
                                            command=lambda: self.open_file(self.set_text_field1))
        self.open_file_button1.grid(column=2, row=0, pady=10, sticky="we")
        self.open_file_button2 = ttk.Button(self.mainframe, text="Mở File thứ hai",
                                            command=lambda: self.open_file(self.set_text_field2))
        self.open_file_button2.grid(column=2, row=2, pady=10, sticky="we")

        self.check_button = ttk.Button(self.mainframe, text="Kiểm tra", command=self.check_text, style='Custom.TButton')
        self.check_button.grid(column=2, row=3, pady=10, sticky="e")
        self.style = ttk.Style()
        self.style.configure('Custom.TButton', font=('Arial', 14))

        self.progress_bar_label1 = None
        self.progress_bar_label2 = None
        self.progress_bar_label3 = None
        self.result_label = None
        self.summary_label = None

        self.root.mainloop()

    def center_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        window_width = 810
        window_height = 500

        x_coordinate = int((screen_width - window_width) / 2)
        y_coordinate = int((screen_height - window_height) / 2)

        self.root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    def save_to_excel(self):
        # Lấy nội dung từ text field 1 và text field 2
        text1 = self.set_text_field1.get("1.0", "end-1c").strip()
        text2 = self.set_text_field2.get("1.0", "end-1c").strip()

        # Lấy giá trị từ result_label và summary_label
        result_value = self.result_label.cget("text")
        summary_value = self.summary_label.cget("text")

        # Tạo DataFrame từ dữ liệu của lần kiểm tra hiện tại
        data = {
            "Văn bản 1": [text1],
            "Văn bản 2": [text2],
            "Kết quả": [summary_value],
            "Kết luận": [result_value]
        }
        df_new = pd.DataFrame(data)

        # Yêu cầu người dùng chọn nơi lưu trữ file Excel
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])

        # Nếu người dùng đã chọn một vị trí lưu trữ
        if file_path:
            try:
                # Kiểm tra xem tệp Excel đã tồn tại hay không
                if os.path.exists(file_path):
                    # Đọc nội dung hiện có của tệp Excel vào DataFrame
                    df_existing = pd.read_excel(file_path)
                    # Thêm dữ liệu từ lần kiểm tra hiện tại vào DataFrame hiện có
                    df_updated = pd.concat([df_existing, df_new], ignore_index=True)
                else:
                    # Nếu tệp Excel không tồn tại, sử dụng DataFrame mới
                    df_updated = df_new

                # Ghi lại DataFrame vào tệp Excel
                df_updated.to_excel(file_path, index=False)
                messagebox.showinfo("Thông báo", "Dữ liệu đã được chèn vào file Excel thành công.")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Có lỗi xảy ra khi ghi vào file Excel: {str(e)}")

    def open_file(self, text_field):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            text_field.delete("1.0", tk.END)  # Xóa nội dung hiện có trong Text
            text_field.insert(tk.END, text)  # Chèn nội dung mới từ file vào Text

    def check_text(self):
        if self.progress_bar_label1:
            self.progress_bar_label1.destroy()
        if self.progress_bar_label2:
            self.progress_bar_label2.destroy()
        if self.progress_bar_label3:
            self.progress_bar_label3.destroy()
        if self.result_label:
            self.result_label.destroy()
        if self.summary_label:
            self.summary_label.destroy()

        self.progress_bar1 = ttk.Progressbar(self.mainframe, orient="horizontal", length=180, mode="determinate")
        self.progress_bar1.grid(column=0, row=4, pady=5, sticky="w")
        self.progress_bar_label1 = ttk.Label(self.mainframe, text="", font=("Arial", 10))
        self.progress_bar_label1.grid(column=0, row=5, pady=5, sticky="w")

        self.progress_bar2 = ttk.Progressbar(self.mainframe, orient="horizontal", length=180, mode="determinate")
        self.progress_bar2.grid(column=0, row=6, pady=5, sticky="w")
        self.progress_bar_label2 = ttk.Label(self.mainframe, text="", font=("Arial", 10))
        self.progress_bar_label2.grid(column=0, row=7, pady=5, sticky="w")

        self.progress_bar3 = ttk.Progressbar(self.mainframe, orient="horizontal", length=180, mode="determinate")
        self.progress_bar3.grid(column=0, row=8, pady=5, sticky="w")
        self.progress_bar_label3 = ttk.Label(self.mainframe, text="", font=("Arial", 10))
        self.progress_bar_label3.grid(column=0, row=9, pady=5, sticky="w")

        text1 = self.set_text_field1.get("1.0", "end-1c").strip()
        text2 = self.set_text_field2.get("1.0", "end-1c").strip()

        if not text1 or not text2:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập đủ văn bản vào cả hai trường.")
            return

        # Tạo và khởi chạy một luồng để thực hiện tính toán
        threading.Thread(target=self.process_checking, args=(text1, text2)).start()

    def process_checking(self, text1, text2):
        # Sửa chính tả
        generated_text_1 = corrector(text1, max_new_tokens=1500)
        generated_text_2 = corrector(text2, max_new_tokens=1500)
        generated_text1 = generated_text_1[0]['generated_text']
        generated_text2 = generated_text_2[0]['generated_text']
        # Dịch đoạn văn bản sang tiếng Việt
        translated_t1 = translator.translate(generated_text1, src='auto', dest='vi').text
        translated_t2 = translator.translate(generated_text2, src='auto', dest='vi').text
        # Tạo vector cho mỗi câu trong đoạn văn bản
        sentence_vectors1 = get_sentence_vectors(translated_t1)
        sentence_vectors2 = get_sentence_vectors(translated_t2)
        # Tính cosine similarity giữa các câu từ hai đoạn văn bản
        average_sentence_similarity = calculate_sentence_similarity(sentence_vectors1, sentence_vectors2)
        # Tính % từ trùng lặp giữa hai văn bản
        overlap_ratio = word_overlapping(translated_t1, translated_t2)
        # Tính % trùng lặp hình thức giữa hai văn bản
        similarity_percentage = text_similarity_percentage(translated_t1, translated_t2)
        # Tính trung bình cộng của các chỉ số tương đồng
        average_similarity = (average_sentence_similarity*0.6 + overlap_ratio*0.3 + similarity_percentage*0.1)
        # Thiết lập giá trị ban đầu của progress bar là 0
        self.progress_bar1["value"] = 0
        self.progress_bar2["value"] = 0
        self.progress_bar3["value"] = 0

        # Cập nhật giá trị của progress bar từ 0% đến giá trị mong muốn
        target_value = average_sentence_similarity * 100
        self.progress_bar_label1.config(text=f"Cosine similarity: {target_value:.2f}%")
        for i in range(int(target_value) + 1):
            self.progress_bar1["value"] = i
            time.sleep(0.01)  # Tạm dừng để tạo hiệu ứng chuyển động
            self.root.update_idletasks()  # Cập nhật giao diện để hiển thị thay đổi của progress bar

        target_value = overlap_ratio * 100
        self.progress_bar_label2.config(text=f"Trùng lặp từ: {target_value:.2f}%")
        for i in range(int(target_value) + 1):
            self.progress_bar2["value"] = i
            time.sleep(0.01)
            self.root.update_idletasks()

        target_value = similarity_percentage * 100
        self.progress_bar_label3.config(text=f"Trùng lặp hình thức: {target_value:.2f}%")
        for i in range(int(target_value) + 1):
            self.progress_bar3["value"] = i
            time.sleep(0.01)

        # Tạo label để hiển thị kết quả
        self.summary_label = ttk.Label(self.mainframe, text=f"Tổng kết: {average_similarity * 100:.2f}%", font=("Arial", 15))
        self.summary_label.grid(column=1, row=4, pady=5, sticky="w")
        self.result_label = ttk.Label(self.mainframe, font=("Arial", 14))

        # Kiểm tra và hiển thị kết quả dựa trên giá trị trung bình
        if average_similarity*100 < 30:
            self.result_label.config(text="Độ tương đồng giữa 2 văn bản thấp", foreground="green")
        else:
            if average_similarity * 100 < 60:
                self.result_label.config(text="Độ tương đồng giữa 2 văn bản trung bình", foreground="orange")
            else:
                self.result_label.config(text="Độ tương đồng giữa 2 văn bản cao", foreground="red")

        # Hiển thị label trên giao diện
        self.result_label.grid(column=1, row=6, pady=5, sticky="w")
        print(generated_text1)
        print(translated_t1)
        print(generated_text2)
        print(translated_t2)

if __name__ == '__main__':
    App()

