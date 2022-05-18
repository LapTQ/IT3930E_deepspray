# TASK: sinh dữ liệu huấn luyện sao cho giống thật

['bag', 'lobe', 'Detached ligament', 'drop', 'Attached ligament']

## Bước 1: Tách ra thành 2 phần: giọt bắn & dòng chính.

### Cách 1: Gán nhãn thủ công cho 1 số giọt.

Cắt bỏ phần thừa của hình chữ nhật đi chỉ lấy giọt

### Cách 2: Dùng kỹ thuật xử lý ảnh để tự động detect và gán nhãn các bouding boxes.

#### 2.1. Tiêu chí
 
1. Giữ được giọt li ti
2. Nhận biết được overlap, phân biệt được ligament vs drop

#### 2.2. Hướng giải quyết

##### 2.2.1. Contour

https://learnopencv.com/contour-detection-using-opencv-python-c/
https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
https://docs.opencv.org/4.x/d1/d32/tutorial_py_contour_properties.html
https://docs.opencv.org/4.x/d5/d45/tutorial_py_contours_more_functions.html

* GLOBAL THRESHOLDING:
  1. Giữ được giọt li ti: 
     * [+] những giọt gần nhau có pixel ở biên nhạt hơn, nên việc thresholding có thể loại bỏ 1 phần sự dính nhau này.
     * [-] nếu threshold value nhỏ hơn thì có nhiều giọt bị to ra, lớn hơn thì mất
     * [-] nếu pixel ở biên vẫn thoát được ngưỡng sẽ thành trắng hết => 2 vật bị nối với nhau (dù là 1 pixel góc) thì cũng được xem là 1 contour
     * **//TODO**: cải thiện thuật toán thresholding hoặc cải thiện chất lượng ảnh (HE chẳng hạn) để không bỏ sót những giọt li ti và loại bỏ nhiều hơn phần biên mờ.
     * **//TODO**: cải thiện thuật toán countour: trường hợp 2 giọt tròn bị dính hoặc overlap thì tự tách contour thành các phương trình đường tròn (VD Hough transform). 
  2. Nhận biết được overlap, phân biệt được ligament vs drop
     * dùng luật về tỉ lệ và diện tích: 
       * [-] nhiều trường hợp các giọt bị dính vào nhau tạo nên tỉ lệ < 3 vẫn được xem là drop. Khả năng là không dùng được.
     * match shape (đang dùng HuMoment) với vài luật (chưa phải match nội dung như SIFT): 
       * [+] phân biệt được drop vs (ligament, dính nhau) khá tốt. Hiện tại phải dùng 2 template (cho giọt tròn và dài) của drop để lọc. Sau khi tách được drop thì lọc bỏ để xử lý dấu [-] sau:
       * [-] chưa biết cách phân biệt ligament vs dính nhau.
         * **//TODO**: kết hợp nhiều tập luật cho các template. Thử dùng mô hình xác suất xem sao.
         * => thử dùng erosion, dilation để lấy mask của vật to, rồi subtract ra vật nhỏ **(CHỊU, KHÔNG CÁCH NÀO BIẾT DIỂM DỪNG)** 
         * => [- chưa tìm thấy hàm] thử dùng luật về độ căng của contour **(CHỊU, CŨNG NHƯ TRÊN)**
       * [-] chưa tách được contour của những giọt hay ligament bị dính vào dòng chính để mà phân tích thêm.
         * => thử dùng erosion, dilation. **(CHỊU, CÓ VẺ THEO CẢM QUAN THÌ CŨNG KHÔNG BIẾT ĐIỂM DỪNG)**
       * [-] không thể áp dụng thresholding này để nhận biết được các đối tượng nằm đè lên dòng chính.
  * => CHỐT LẠI: theo hướng THRESHOLDING này thỉ chỉ có thể tận dụng ở bước THRESHOLD -> HU MOMENT để lọc ra giọt li ti và các drop nằm tách biệt hoàn toàn. 
* Kế thừa kết quả của Global thresholding: Cải thiện chất lượng ảnh gốc rồi dùng contour trên ảnh gốc, hoặc trên ảnh edge (HIỆN TẠI chưa thể dùng được Hough, edge cũng rối quá không lấy được contour. Chắc do chưa biết dùng =_=)
* Kế thừa kết quả của GLobal thresholding: Cải thiện chất lượng ảnh gốc rồi dùng SIFT, SURF, KAZE, AKAZE, ORB, and BRISK. (Ảnh toàn màu xanh thế này thì...)

Chắc là phải theo cái hướng này:
- Tìm cách nào đó để detect bằng cách truyền thống các giọt nhỏ và liti có kích thước bé đến mức mà DL hiện tại không thể làm được.
- Lọc bỏ những cái đó đi trên ảnh gốc
- Lấy những đối tượng đã gán nhãn được trong mấy tuần thử nghiệm trên và dán vào ảnh khác. (Lọc lấy dòng chính trên 1 ảnh, làm lu mờ hết các giọt đè lên nó. Với các attached ligament thì lấy các deattached ligament, cắt bỏ phần cuối đi rồi dán vào dòng chính.) 
     
Chú ý: sample 1 phần các contour đã detect được để gán nhãn (giảm số lượng gán sai) và chọn được cái nào thì xóa khỏi ảnh gốc cái đó để xử lý phần còn lại.
      
 

* Dùng edge detection:
  1. Giữ được giọt li ti:
       - [-] sau khi lấy cạnh: nhiều giọt li ti có diện tích gần bằng giọt to hơn 
       - [-] sau khi lấy cạnh: một số giọt bị dính vào nhau
       - [-] sau khi lấy cạnh: nhiều giọt dính nhau bị mất biên
       - Nếu dùng bộ lọc blur trước khi lấy cạnh: giải quyết [- 1, 2], nhưng làm trầm trọng thêm [- 3], và làm mất đi nhiều giọt li ti
     
  2. Nhận biết được overlap (Tương tự như trên)
      
 
contour phụ thuộc vào xử lí phía trước.
canny có thể làm thiếu vài pixel = > contour không kín
canny có thể loại được nhiễu



   

##### 2.2.2 Connected component


## Bước 2: Dán các giọt bắn (đã có gán nhãn) vào dòng chính.
Các giọt bắn có thể:
* được xoay ở vị trí cố định (pending)
* thay đổi kích thước (pending)
* đổi vị trí cho nhau => lợi ích: các giọt có kích thước khác nhau nên lúc paste có thể sẽ tạo ra overlap (OK)
* xê dịch trong 1 vùng nhất định (pending)
* tương tự, với dòng chính của ảnh khác có cùng thông số hình học với dòng chính (pending)

Chú ý:
* kích thước giọt bắn khi resize phải hợp lí, tốt nhất là không được thay đổi phân bố của diện tích (pending)
* loại bỏ background xuang quanh chỉ giữ nguyên phần giọt của mỗi box thôi (working)
* khi cắt các giọt bị trùng với dòng chính thì sẽ để lại khoảng màu trắng ở dòng chính, phải làm sao? (working)
* khi dán các giọt vào ảnh mới, thì phải dán ở 1 vị trí hợp lý (không nói trường hợp dán vào vị trí cũ của 1 giọt khác), chắc phải tìm hiểu phân bố vị trí trong không gian ảnh






Segmenting binary object:
- boundary detection
https://learnopencv.com/edge-detection-using-opencv/
https://pyimagesearch.com/2019/03/04/holistically-nested-edge-detection-with-opencv-and-deep-learning/

- connected component
- SIFT

- hỗ trợ bởi
+ xử lí pixel, filter, fft,

+ dilate hoặc erose




contour:
+ detect được giọt rất nhỏ
- không xử lý được overlap
DNN:
+ xử lý được overlap
- không detect đc giọt rất nhỏ


Generative models:

https://www.tensorflow.org/tutorials/generative/cvae
https://www.tensorflow.org/tutorials/generative/dcgan
https://keras.io/examples/generative/vae/
https://www.kaggle.com/code/theblackmamba31/generating-fake-faces-using-gan/notebook
https://blog.paperspace.com/face-generation-with-dcgans/
https://towardsdatascience.com/generating-with-style-the-mechanics-behind-nvidias-highly-realistic-gan-images-b6937237e3c6
http://myreadersspace.com/2020/08/24/stylegan-2-a-better-version-of-stylegan/

## References

https://colab.research.google.com/drive/1VdPd3ejA8hWiLlZSgx7L0rAAAqyROH3t?usp=sharing&fbclid=IwAR10UUbo3_ykmJVx7_5ReJ3kgNhaNgS4i_n2GA7AFEfUsaeSB43pg4UpkCg
https://husteduvn-my.sharepoint.com/personal/sang_dinhviet_hust_edu_vn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsang%5Fdinhviet%5Fhust%5Fedu%5Fvn%2FDocuments%2F%21DATA%2FDeepSrayPackage%2FPACIFIC%2D20210309T043014Z%2D001%2FPACIFIC%2Fiso%5Fpng%20%281%29%2FDeepSpray%20Data&ga=1
https://github.com/sangdv/deepspray
https://colab.research.google.com/drive/1UDU6gRinL8c6hRSuLG1mNzeNdTpRsOaa?usp=sharing&fbclid=IwAR0CfuttFANX3rdQJkUQknpuqQSYCHNobqNxudQbnWs9wTr4MT04eHGL_6I