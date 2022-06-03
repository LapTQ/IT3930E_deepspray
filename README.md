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
         * **//TODO**: kết hợp nhiều tập luật cho các template (thử dùng mô hình xác suất xem sao), và cải thiện chất kuownjg thresholding để giọt tròn được tròn hơn
         * => thử dùng erosion, dilation để lấy mask của vật to, rồi subtract ra vật nhỏ **(CHỊU, KHÔNG CÁCH NÀO BIẾT DIỂM DỪNG)** 
         * => [- chưa tìm thấy hàm] thử dùng luật về độ căng của contour **(CHỊU, CŨNG NHƯ TRÊN)**
       * [-] chưa tách được contour của những giọt hay ligament bị dính vào dòng chính để mà phân tích thêm.
         * => thử dùng erosion, dilation. **(CHỊU, CÓ VẺ THEO CẢM QUAN THÌ CŨNG KHÔNG BIẾT ĐIỂM DỪNG)**
       * [-] không thể áp dụng thresholding này để nhận biết được các đối tượng nằm đè lên dòng chính.
  * => **CHỐT LẠI**: theo hướng THRESHOLDING này thỉ chỉ có thể tận dụng ở bước THRESHOLD -> HU MOMENT để lọc ra giọt li ti và các drop nằm tách biệt hoàn toàn. Phần còn lại là: Dòng chính và những thứ bị dính vào nó, ligament, các giọt dính nhau hoặc đè nhau.
* Kế thừa kết quả của Global thresholding: Cải thiện chất lượng ảnh gốc rồi dùng contour trên ảnh gốc, hoặc trên ảnh edge (HIỆN TẠI chưa thể dùng được Hough, edge cũng rối quá không lấy được contour. Chắc do chưa biết dùng =_=. HOUGH phụ thuộc hoàn toàn vào Canny, nên có thể phải code lại Canny.)
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




## References

https://colab.research.google.com/drive/1VdPd3ejA8hWiLlZSgx7L0rAAAqyROH3t?usp=sharing&fbclid=IwAR10UUbo3_ykmJVx7_5ReJ3kgNhaNgS4i_n2GA7AFEfUsaeSB43pg4UpkCg
https://husteduvn-my.sharepoint.com/personal/sang_dinhviet_hust_edu_vn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsang%5Fdinhviet%5Fhust%5Fedu%5Fvn%2FDocuments%2F%21DATA%2FDeepSrayPackage%2FPACIFIC%2D20210309T043014Z%2D001%2FPACIFIC%2Fiso%5Fpng%20%281%29%2FDeepSpray%20Data&ga=1
https://github.com/sangdv/deepspray
https://colab.research.google.com/drive/1UDU6gRinL8c6hRSuLG1mNzeNdTpRsOaa?usp=sharing&fbclid=IwAR0CfuttFANX3rdQJkUQknpuqQSYCHNobqNxudQbnWs9wTr4MT04eHGL_6I


https://www.elveflow.com/microfluidic-reviews/droplet-digital-microfluidics/droplet-detection-measurement-microfluidic-channels/
các thuật toán tracking có thể giúp ích gì không? Biết đâu có thể track được giọt di chuyển phía sau nhau
nếu dựng lại backbone để yolo detect vật nhỏ được không (phải hiểu nguyên lý của mạng CNN). Tại sao không thiết kế backbone theo dạng Unet? Liệu có phải chỉ vì vấn đề chi phí tính toán?

 |- drop
      |- liti (CV)
      |    |- tách biệt (-> dùng luật về diện tích có thể bắt được hết)
      |    |- dính nhau hoặc dính vào vật khác (-> 0.01%, quá bé nên không thể phân biệt được với bóng mờ ở viền)
      |    |- nằm đè lên vật khác (rất khó xác định do chỉ là một chấm nhỏ màu hơi đậm hơn): [A new algorithm for detecting and correcting bad pixels in infrared images](http://www.scielo.org.co/scielo.php?pid=S0120-56092010000200020&script=sci_arttext&tlng=en) 
      |- bé (CV)
      |    |- tách biệt (-> dùng Humoment sẽ bắt được gần hết, không thể dùng luật tỉ lệ box vì nó không thể phân biệt được với trường hợp nhiều giọt bị dính nhau sinh ra box tỉ lệ vuông)
      |    |- dính nhau hoặc dính vào vật khác
      |    |- overlap với nhau hoặc với vật khác (contour có thể nhầm với ligament)
      |    |- nằm đè lên vật khác (khó xác định do đường biên mờ)
      |- trung bình (DL, sinh dữ liệu bằng CV)      -> CV detect một vài giọt trung bình thật, hoặc tăng kích thước các giot bé và deform?
      |    |- ... (giống drop bé)
      |- to (DL, sinh dữ liệu từ CV)                -> CV detect một vài giọt to thật, hoặc tăng kích thước các giọt bé (có thể kết hợp các giọt bé + xóa biên) và deform?
      |    |- ... (giống drop trung bình)
 |- deattached liagement (contour có thể nhầm với các drop dính nhau) (DL, sinh dữ liệu từ CV) -> CV detect một vài ligament thật, hoặc thăng kích thước của drop (có thể kết hợp nhiều giọt và làm mượt) và deform?
      |- tách biệt
      |- dính vào các vật khác
      |- nằm đè lên dòng chính
      |- bị vật khác đè lên một/nhiều phần hoặc overlap (bị mất đường biên nếu bị đè lên 1 phần)
 |- attached ligament (DL, sinh dữ liệu từ CV)     -> CV không thể detect thật, nên có thể biến đổi deattached ligament
      |- khác với deattached ligament ở chỗ: 1 phần đầu không có biên mà sẽ hòa vào dòng chính
      |- gần như luôn bị vật khác đè lên tạo ra biên gây nhầm lẫn với deattached ligament
 |- bag
 |- lobe

